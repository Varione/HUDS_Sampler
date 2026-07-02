"""Synthetic data generators for HUDS active learning benchmarks.

Generates labeled candidate pools (pandas DataFrames) for three model types:
vector-to-vector, vector-to-time-series, and vector-to-image.

Each generator returns a DataFrame compatible with the HUDS training pipeline:
  sample_id  x_0  ...  x_{D-1}  <output_columns...>
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_v2v(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate vector-to-vector labeled data.

    Each output is a nonlinear combination of input features using sin, cos,
    and interaction terms so the surrogate model faces a nontrivial regression task.

    Args:
        n_samples: Number of samples to generate.
        input_dim: Dimensionality of input space (number of x columns).
        output_dim: Dimensionality of output space (number of y columns).
        noise: Standard deviation of additive Gaussian noise on outputs.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [sample_id, x_0, ..., x_{D-1}, y_0, ..., y_{K-1}].
    """
    rng = np.random.default_rng(seed)

    # Input: uniform in [0, 1]
    X = rng.uniform(0.0, 1.0, size=(n_samples, input_dim))

    # Build output with nonlinear transforms per output dimension
    Y = np.zeros((n_samples, output_dim), dtype=np.float64)
    for k in range(output_dim):
        # Shift features so each output sees a different nonlinear landscape
        phase = float(k) * np.pi / max(1, output_dim - 1)

        sin_term = np.sin(2.0 * np.pi * X[:, 0 % input_dim] + phase)
        cos_term = np.cos(2.0 * np.pi * X[:, 1 % input_dim] + phase)

        # Interaction terms between pairs of inputs
        i_a = (k) % input_dim
        i_b = (k + 1) % input_dim
        interaction = X[:, i_a] * X[:, i_b]

        # Quadratic term
        quad = X[:, (k + 2) % input_dim] ** 2

        Y[:, k] = sin_term + cos_term + 0.5 * interaction + 0.3 * quad

    # Add Gaussian noise
    Y += rng.normal(0.0, noise, size=Y.shape)

    # Build DataFrame
    data = {"sample_id": np.arange(1, n_samples + 1)}
    for d in range(input_dim):
        data[f"x_{d}"] = X[:, d]
    for k in range(output_dim):
        data[f"y_{k}"] = Y[:, k]

    return pd.DataFrame(data)


def generate_v2ts(
    n_samples: int,
    input_dim: int,
    seq_len: int = 50,
    output_dim: int = 2,
    noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate vector-to-time-series labeled data.

    Each output channel is a damped oscillation whose parameters (amplitude,
    frequency, damping rate) are determined by the input vector X. This simulates
    a physical system response parameterized by design variables.

    Columns are flattened as:
        y_0_t0, y_0_t1, ..., y_0_t{S-1}, y_1_t0, ..., y_{K-1}_t{S-1}
    matching the HUDS training pipeline which expects one column per output name.

    Args:
        n_samples: Number of samples to generate.
        input_dim: Dimensionality of input space.
        seq_len: Length of each output time series.
        output_dim: Number of independent output channels.
        noise: Standard deviation of additive Gaussian noise on outputs.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [sample_id, x_0..x_{D-1}, y_0_t0..y_{K-1}_t{S-1}].
    """
    rng = np.random.default_rng(seed)

    # Input: uniform in [0, 1]
    X = rng.uniform(0.0, 1.0, size=(n_samples, input_dim))

    # Time axis shared across all samples
    t = np.linspace(0.0, 1.0, seq_len)

    # Output shape: (n_samples, output_dim, seq_len)
    Y = np.zeros((n_samples, output_dim, seq_len), dtype=np.float64)

    for k in range(output_dim):
        # Amplitude from first input feature, scaled to [0.5, 2.0]
        amplitude = 0.5 + X[:, 0 % input_dim] * 1.5

        # Frequency from second feature, scaled to [1, 10] Hz
        frequency = 1.0 + X[:, 1 % input_dim] * 9.0

        # Damping rate from third feature, scaled to [0, 10]
        damping = X[:, 2 % input_dim] * 10.0

        # Phase offset from fourth feature
        phase = X[:, 3 % input_dim] * 2.0 * np.pi

        # Damped oscillation: A * exp(-alpha * t) * sin(2*pi*f*t + phi)
        Y[:, k, :] = (
            amplitude[:, np.newaxis]
            * np.exp(-damping[:, np.newaxis] * t[np.newaxis, :])
            * np.sin(2.0 * np.pi * frequency[:, np.newaxis] * t[np.newaxis, :] + phase[:, np.newaxis])
        )

    # Add Gaussian noise
    Y += rng.normal(0.0, noise, size=Y.shape)

    # Flatten to (n_samples, output_dim * seq_len)
    Y_flat = Y.reshape(n_samples, -1)

    # Build DataFrame
    data = {"sample_id": np.arange(1, n_samples + 1)}
    for d in range(input_dim):
        data[f"x_{d}"] = X[:, d]
    for k in range(output_dim):
        for s in range(seq_len):
            data[f"y_{k}_t{s}"] = Y_flat[:, k * seq_len + s]

    return pd.DataFrame(data)


def generate_v2i(
    n_samples: int,
    input_dim: int,
    img_h: int = 32,
    img_w: int = 32,
    channels: int = 1,
    noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate vector-to-image labeled data.

    Each output channel is a 2D field constructed by combining Gaussian radial
    basis functions whose centers, widths, and amplitudes are determined by X.
    This simulates spatial field generation parameterized by design variables.

    The image is flattened in channel-major order (C, H, W) to columns:
        img_0, img_1, ..., img_{H*W*C-1}

    Args:
        n_samples: Number of samples to generate.
        input_dim: Dimensionality of input space.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        channels: Number of output channels.
        noise: Standard deviation of additive Gaussian noise on outputs.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [sample_id, x_0..x_{D-1}, img_0..img_{H*W*C-1}].
    """
    rng = np.random.default_rng(seed)

    # Input: uniform in [0, 1]
    X = rng.uniform(0.0, 1.0, size=(n_samples, input_dim))

    # Spatial grid normalized to [0, 1]
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, img_h),
        np.linspace(0.0, 1.0, img_w),
        indexing="ij",
    )

    # Number of RBFs per channel
    n_rbf = 3

    # Output shape: (n_samples, channels, img_h, img_w)
    Y = np.zeros((n_samples, channels, img_h, img_w), dtype=np.float64)

    for c in range(channels):
        for r in range(n_rbf):
            # RBF center from input features
            cx_idx = (c * n_rbf + r) % input_dim
            cy_idx = (c * n_rbf + r + 1) % input_dim
            cx = X[:, cx_idx]
            cy = X[:, cy_idx]

            # Width from input feature, scaled to [0.05, 0.3]
            width_idx = (c * n_rbf + r + 2) % input_dim
            width = 0.05 + X[:, width_idx] * 0.25

            # Amplitude from input feature, scaled to [0.3, 1.0]
            amp_idx = (c * n_rbf + r + 3) % input_dim
            amplitude = 0.3 + X[:, amp_idx] * 0.7

            # Compute RBF: A * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
            dist_sq = (
                (xx[np.newaxis, :, :] - cx[:, np.newaxis, np.newaxis]) ** 2
                + (yy[np.newaxis, :, :] - cy[:, np.newaxis, np.newaxis]) ** 2
            )
            Y[:, c, :, :] += (
                amplitude[:, np.newaxis, np.newaxis]
                * np.exp(-dist_sq / (2.0 * width[:, np.newaxis, np.newaxis] ** 2))
            )

    # Add Gaussian noise
    Y += rng.normal(0.0, noise, size=Y.shape)

    # Flatten to (n_samples, channels * img_h * img_w)
    total_pixels = channels * img_h * img_w
    Y_flat = Y.reshape(n_samples, -1)

    # Build DataFrame
    data = {"sample_id": np.arange(1, n_samples + 1)}
    for d in range(input_dim):
        data[f"x_{d}"] = X[:, d]
    for p in range(total_pixels):
        data[f"img_{p}"] = Y_flat[:, p]

    return pd.DataFrame(data)
