"""Tests for ResidualMLP model."""

import pytest
import torch
import torch.nn as nn

from huds_app.config import AppConfig, VariableConfig, ModelConfig
from huds_app.model import ResidualMLP, build_model


@pytest.fixture
def config():
    """Two-variable, two-output config with 64 hidden and 2 residual blocks."""
    return AppConfig(
        random_seed=42,
        variables=[
            VariableConfig(name="x", min=0.0, max=1.0, sample_points=5),
            VariableConfig(name="y", min=0.0, max=2.0, sample_points=3),
        ],
        model=ModelConfig(output_names=["fx", "fy"], hidden_dim=64, residual_blocks=2, dropout=0.1),
    )


class TestResidualMLPShape:
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


class TestEdgeCases:
    def test_zero_residual_blocks_works(self, config):
        zero_cfg = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            model=ModelConfig(output_names=["fx"], hidden_dim=32, residual_blocks=0),
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
    def test_build_model_creates_correct_dims(self, config):
        model = build_model(config)
        assert isinstance(model, ResidualMLP)
        assert model.input_layer.in_features == len(config.variables)
        assert model.output_layer.out_features == len(config.model.output_names)

    def test_build_model_has_expected_residual_blocks(self, config):
        model = build_model(config)
        assert len(model.residual_blocks) == config.model.residual_blocks
