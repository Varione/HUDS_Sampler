"""Tests for multi-architecture surrogate models."""

import pytest
import torch
import torch.nn as nn

from huds_app.core.config import AppConfig, ModelConfig, VariableConfig
from huds_app.model.architecture import (
    VectorToVector,
    VectorToTimeSeries,
    VectorToImage,
    build_model,
)


@pytest.fixture
def config():
    """Two-variable, two-output config with 64 hidden and 2 encoder blocks."""
    return AppConfig(
        random_seed=42,
        variables=[
            VariableConfig(name="x", min=0.0, max=1.0, sample_points=5),
            VariableConfig(name="y", min=0.0, max=2.0, sample_points=3),
        ],
        model=ModelConfig(
            output_names=["fx", "fy"],
            hidden_dim=64,
            encoder_blocks=2,
            dropout=0.1,
        ),
    )


class TestVectorToVector:
    def test_forward_returns_correct_shape(self, config):
        model = build_model(config)
        x = torch.randn(32, len(config.variables))
        out = model(x)
        assert out.shape == (32, len(config.model.output_names))

    def test_return_features_returns_tuple(self, config):
        model = build_model(config)
        x = torch.randn(16, len(config.variables))
        result = model(x, return_features=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_feature_dimension_equals_hidden_dim(self, config):
        model = build_model(config)
        x = torch.randn(8, len(config.variables))
        out, features = model(x, return_features=True)
        assert features.shape[1] == config.model.hidden_dim


class TestVectorToTimeSeries:
    def test_build_from_config(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_time_series",
                output_names=[f"ts_{i}_0" for i in range(20)],  # seq_len=20, channels=1
                hidden_dim=32,
                encoder_blocks=1,
                seq_len=20,
                decoder_layers=2,
            ),
        )
        model = build_model(cfg)
        assert isinstance(model, VectorToTimeSeries)

    def test_forward_shape(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_time_series",
                output_names=[f"ts_{i}_{c}" for i in range(10) for c in range(2)],  # seq_len=10, channels=2
                hidden_dim=32,
                encoder_blocks=1,
                seq_len=10,
                decoder_layers=1,
            ),
        )
        model = build_model(cfg)
        x = torch.randn(4, 1)
        out = model(x)
        assert out.shape == (4, 10, 2)

    def test_return_features_shape(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_time_series",
                output_names=[f"ts_{i}_0" for i in range(10)],  # seq_len=10, channels=1
                hidden_dim=32,
                encoder_blocks=1,
                seq_len=10,
                decoder_layers=1,
            ),
        )
        model = build_model(cfg)
        x = torch.randn(4, 1)
        out, features = model(x, return_features=True)
        assert features.shape == (4, 32)


class TestVectorToImage:
    def test_build_from_config(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_image",
                output_names=["field"],
                hidden_dim=64,
                encoder_blocks=1,
                img_h=16,
                img_w=16,
                channels=1,
                decoder_blocks=3,
            ),
        )
        model = build_model(cfg)
        assert isinstance(model, VectorToImage)

    def test_forward_shape(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_image",
                output_names=["temp"],
                hidden_dim=64,
                encoder_blocks=1,
                img_h=16,
                img_w=16,
                channels=1,
                decoder_blocks=3,
            ),
        )
        model = build_model(cfg)
        x = torch.randn(4, 1)
        out = model(x)
        assert out.shape[0] == 4
        assert len(out.shape) == 4

    def test_return_features_shape(self):
        cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(
                model_type="vector_to_image",
                output_names=["temp"],
                hidden_dim=64,
                encoder_blocks=1,
                img_h=16,
                img_w=16,
                channels=1,
                decoder_blocks=3,
            ),
        )
        model = build_model(cfg)
        x = torch.randn(4, 1)
        out, features = model(x, return_features=True)
        assert features.shape == (4, 64)


class TestEdgeCases:
    def test_zero_encoder_blocks_works(self):
        zero_cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(output_names=["fx"], hidden_dim=32, encoder_blocks=0),
        )
        model = build_model(zero_cfg)
        x = torch.randn(10, 1)
        out = model(x)
        assert out.shape == (10, 1)

    def test_forward_on_cpu(self, config):
        """Model forward pass on CPU succeeds regardless of training.device."""
        model = build_model(config)
        model.cpu()
        x = torch.randn(64, len(config.variables))
        out = model(x)
        assert not out.is_cuda


class TestBuildModel:
    def test_build_model_creates_correct_type(self, config):
        model = build_model(config)
        assert isinstance(model, VectorToVector)

    def test_build_model_has_encoder(self, config):
        model = build_model(config)
        assert hasattr(model, "encoder")
        assert len(model.encoder.blocks) == config.model.encoder_blocks
