"""Multi-architecture surrogate models with unified interface.

Three architectures share a common ResidualEncoder backbone:

  Vector-to-Vector (V2V):  Encoder → Linear head
  Vector-to-TimeSeries:    Encoder → GRU decoder → per-step head
  Vector-to-Image:         Encoder → Latent reshape → TransposedConv decoder

All models support return_features=True to expose the encoder embedding
for MC-Dropout uncertainty estimation in HUDS sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared encoder backbone
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Single residual block: Linear → ReLU → Dropout → Linear + skip."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return x + out


class ResidualEncoder(nn.Module):
    """Shared encoder: input_dim → hidden_dim with N residual blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# Architecture 1: Vector-to-Vector (original ResidualMLP)
# ---------------------------------------------------------------------------

class VectorToVector(nn.Module):
    """Vector input → Vector output.

    Encoder (residual MLP) → Linear head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        encoder_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(input_dim, hidden_dim, encoder_blocks, dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        output = self.head(features)
        if return_features:
            return output, features
        return output


# ---------------------------------------------------------------------------
# Architecture 2: Vector-to-TimeSeries
# ---------------------------------------------------------------------------

class TimeSeriesDecoder(nn.Module):
    """GRU-based sequence decoder.

    Takes a single encoder embedding, repeats it for seq_len steps,
    runs a GRU, and projects each step to output_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding: (batch, hidden_dim) → (batch, seq_len, hidden_dim)
        seq_input = embedding.unsqueeze(1).expand(-1, self.seq_len, -1)
        gru_out, _ = self.gru(seq_input)
        return self.head(gru_out)  # (batch, seq_len, output_dim)


class VectorToTimeSeries(nn.Module):
    """Vector input → Time-series output.

    Encoder (residual MLP) → GRU decoder → per-step projection.
    Output shape: (batch, seq_len, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        encoder_blocks: int,
        dropout: float = 0.1,
        seq_len: int = 50,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(input_dim, hidden_dim, encoder_blocks, dropout)
        self.decoder = TimeSeriesDecoder(hidden_dim, output_dim, seq_len, decoder_layers)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        output = self.decoder(features)
        if return_features:
            return output, features
        return output


# ---------------------------------------------------------------------------
# Architecture 3: Vector-to-Image
# ---------------------------------------------------------------------------

class TransposedConvBlock(nn.Module):
    """Single upsampling block: ConvTranspose2d → BatchNorm → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ImageDecoder(nn.Module):
    """Multi-stage transposed-conv decoder with size restoration.

    Reshapes a 1D embedding into a small latent feature map and
    progressively upsamples, then interpolates to target (img_h, img_w).

    Architecture:
        Linear(hidden_dim -> latent_ch * latent_h * latent_w)
        -> Reshape(latent_ch, latent_h, latent_w)
        -> [TransposedConvBlock] * num_blocks  (each doubles spatial size)
        -> Interpolate to target size
        -> Conv2d(latent_ch // 2^(num_blocks-1) -> channels, 1)

    FIX 6: Ensures output size matches img_h x img_w by interpolating after decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        channels: int,
        img_h: int,
        img_w: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        # Each transposed conv block doubles spatial dimensions.
        latent_h = max(1, img_h >> num_blocks)
        latent_w = max(1, img_w >> num_blocks)

        # Start with a reasonable latent channel count derived from hidden_dim.
        # Must be divisible by 2^(num_blocks-1) for clean halving.
        base_ch = max(channels, hidden_dim // (latent_h * latent_w))
        # Round up to nearest power-of-2 multiple for clean division
        import math

        min_multiple = 1 << (num_blocks - 1) if num_blocks > 1 else 1
        start_ch = ((base_ch + min_multiple - 1) // min_multiple) * min_multiple
        start_ch = max(start_ch, channels)

        self.latent_shape = (start_ch, latent_h, latent_w)
        actual_flat = start_ch * latent_h * latent_w
        self.flatten_proj = nn.Linear(hidden_dim, actual_flat)

        # Build decoder: each block halves channels and doubles spatial size.
        layers: list[nn.Module] = []
        prev_ch = start_ch
        for i in range(num_blocks):
            next_ch = max(prev_ch // 2, channels)
            layers.append(TransposedConvBlock(prev_ch, next_ch))
            prev_ch = next_ch

        self.blocks = nn.Sequential(*layers)
        # Final 1x1 conv to exact channel count.
        self.final_conv = nn.Conv2d(prev_ch, channels, 1)

        # FIX 6: Store target size for interpolation after decoder
        self.img_h = img_h
        self.img_w = img_w

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        b = embedding.shape[0]
        x = self.flatten_proj(embedding)
        x = x.view(b, *self.latent_shape)
        x = self.blocks(x)
        x = self.final_conv(x)

        # FIX 6: Ensure output size matches target by interpolation
        if (x.shape[2], x.shape[3]) != (self.img_h, self.img_w):
            x = F.interpolate(
                x,
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            )

        return x


class VectorToImage(nn.Module):
    """Vector input → 2D image output.

    Encoder (residual MLP) → Latent reshape → TransposedConv decoder.
    Output shape: (batch, channels, img_h, img_w)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_blocks: int,
        dropout: float = 0.1,
        img_h: int = 32,
        img_w: int = 32,
        channels: int = 1,
        decoder_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(input_dim, hidden_dim, encoder_blocks, dropout)
        self.decoder = ImageDecoder(hidden_dim, channels, img_h, img_w, decoder_blocks)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        output = self.decoder(features)
        if return_features:
            return output, features
        return output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config) -> nn.Module:
    """Construct a model from AppConfig.

    Reads model.model_type to dispatch to the correct architecture.
    For V2TS, output_dim = len(output_names) / seq_len (num channels).
    For V2I, channels come directly from config.
    For V2V, output_dim = len(output_names).
    """
    input_dim = len(config.variables)
    hidden_dim = config.model.hidden_dim
    encoder_blocks = config.model.encoder_blocks
    dropout = config.model.dropout
    model_type = config.model.model_type

    if model_type == "vector_to_time_series":
        seq_len = config.model.seq_len
        output_dim = len(config.model.output_names) // seq_len
        return VectorToTimeSeries(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            encoder_blocks=encoder_blocks,
            dropout=dropout,
            seq_len=seq_len,
            decoder_layers=config.model.decoder_layers,
        )

    if model_type == "vector_to_image":
        return VectorToImage(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            encoder_blocks=encoder_blocks,
            dropout=dropout,
            img_h=config.model.img_h,
            img_w=config.model.img_w,
            channels=config.model.channels,
            decoder_blocks=config.model.decoder_blocks,
        )

    # Default: vector_to_vector
    output_dim = len(config.model.output_names)
    return VectorToVector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        encoder_blocks=encoder_blocks,
        dropout=dropout,
    )


# Backward-compatibility alias
ResidualMLP = VectorToVector
