"""Tests for config loading and validation."""

import json
import tempfile

import pytest

from huds_app.core.config import (
    AppConfig,
    CandidatePoolConfig,
    HUDSConfig,
    ModelConfig,
    TrainingConfig,
    VariableConfig,
    ValidationConfig,
    load_config,
    validate_config,
)


def _valid_config_dict() -> dict:
    """Return a minimal valid config dictionary."""
    return {
        "project_name": "test",
        "random_seed": 42,
        "variables": [
            {"name": "x1", "min": 0.0, "max": 10.0, "sample_points": 5},
        ],
        "candidate_pool": {
            "total_samples": 100,
            "train_ratio": 0.8,
            "validation_ratio": 0.2,
        },
        "model": {"output_names": ["y"], "hidden_dim": 64},
    }


class TestLoadConfig:
    def test_load_valid_config(self):
        cfg = _valid_config_dict()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        result = load_config(path)
        assert isinstance(result, AppConfig)
        assert result.project_name == "test"
        assert len(result.variables) == 1
        assert result.variables[0].name == "x1"

    def test_load_variable_unit(self):
        cfg = _valid_config_dict()
        cfg["variables"][0]["unit"] = "mm"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        result = load_config(path)

        assert result.variables[0].unit == "mm"


class TestValidateConfigDuplicateNames:
    def test_duplicate_variable_names_raises(self):
        cfg = _valid_config_dict()
        cfg["variables"].append({"name": "x1", "min": 0.0, "max": 5.0, "sample_points": 3})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="unique"):
            load_config(path)


class TestValidateConfigMinGeMax:
    def test_min_ge_max_raises(self):
        cfg = _valid_config_dict()
        cfg["variables"][0]["min"] = 10.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="min < max"):
            load_config(path)


class TestValidateConfigEmptyVariables:
    def test_empty_variables_raises(self):
        cfg = _valid_config_dict()
        cfg["variables"] = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="at least one"):
            load_config(path)


class TestValidateConfigRatioSum:
    def test_invalid_ratio_raises(self):
        cfg = _valid_config_dict()
        cfg["candidate_pool"]["train_ratio"] = 0.7
        cfg["candidate_pool"]["validation_ratio"] = 0.2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="must equal 1.0"):
            load_config(path)


class TestValidateConfigTotalSamples:
    def test_total_samples_zero_raises(self):
        cfg = _valid_config_dict()
        cfg["candidate_pool"]["total_samples"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="total_samples"):
            load_config(path)


class TestValidateConfigHiddenDim:
    def test_hidden_dim_zero_raises(self):
        cfg = _valid_config_dict()
        cfg["model"]["hidden_dim"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="hidden_dim"):
            load_config(path)


class TestValidateConfigTopPThreshold:
    def test_top_p_threshold_zero_raises(self):
        cfg = _valid_config_dict()
        cfg["huds"] = {"use_top_p": True, "top_p_threshold": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="top_p_threshold"):
            load_config(path)

    def test_top_p_threshold_over_one_raises(self):
        cfg = _valid_config_dict()
        cfg["huds"] = {"use_top_p": True, "top_p_threshold": 1.5}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        with pytest.raises(ValueError, match="top_p_threshold"):
            load_config(path)

    def test_top_p_disabled_allows_any_threshold(self):
        cfg = _valid_config_dict()
        cfg["huds"] = {"use_top_p": False, "top_p_threshold": 1.5}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            path = f.name

        result = load_config(path)
        assert result.huds.use_top_p is False
