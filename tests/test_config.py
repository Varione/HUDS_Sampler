"""Tests for config loading and validation."""

import json
import tempfile

import pytest

from huds_app.core.config import (
    AppConfig,
    CandidatePoolConfig,
    HUDSConfig,
    MAXWELL_UNIT_PRESETS,
    ModelConfig,
    TrainingConfig,
    VariableConfig,
    ValidationConfig,
    load_config,
    resolve_maxwell_unit,
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


class TestMaxwellUnitPresets:
    """Test Maxwell unit preset resolution."""

    def test_exact_unit_string_passthrough(self):
        """Exact unit string (mm, Hz, A) should pass through unchanged."""
        assert resolve_maxwell_unit("mm") == "mm"
        assert resolve_maxwell_unit("Hz") == "Hz"
        assert resolve_maxwell_unit("A") == "A"
        assert resolve_maxwell_unit("km_per_hour") == "km_per_hour"

    def test_preset_name_resolution(self):
        """Preset names should resolve to Maxwell unit strings."""
        assert resolve_maxwell_unit("millimeter") == "mm"
        assert resolve_maxwell_unit("hertz") == "Hz"
        assert resolve_maxwell_unit("ampere") == "A"
        assert resolve_maxwell_unit("kmh") == "km_per_hour"
        assert resolve_maxwell_unit("mps") == "m_per_sec"

    def test_case_insensitive_lookup(self):
        """Preset lookup should be case-insensitive."""
        assert resolve_maxwell_unit("MM") == "mm"
        assert resolve_maxwell_unit("MILLIMETER") == "mm"
        assert resolve_maxwell_unit("Hertz") == "Hz"
        assert resolve_maxwell_unit("KM_PER_HOUR") == "km_per_hour"

    def test_empty_string_returns_empty(self):
        """Empty string should return empty."""
        assert resolve_maxwell_unit("") == ""
        assert resolve_maxwell_unit("  ") == "  "

    def test_unknown_unit_passthrough(self):
        """Unknown unit strings should pass through unchanged."""
        assert resolve_maxwell_unit("custom_unit") == "custom_unit"
        assert resolve_maxwell_unit("rad/s") == "rad/s"

    def test_variable_config_resolved_unit(self):
        """VariableConfig.resolved_unit() should apply preset resolution."""
        v = VariableConfig(name="gap", min=0, max=10, sample_points=5, unit="millimeter")
        assert v.resolved_unit() == "mm"

        v2 = VariableConfig(name="freq", min=1, max=100, sample_points=10, unit="hertz")
        assert v2.resolved_unit() == "Hz"

        v3 = VariableConfig(name="speed", min=0, max=50, sample_points=10, unit="kmh")
        assert v3.resolved_unit() == "km_per_hour"

    def test_all_maxwell_presets_defined(self):
        """Verify all expected Maxwell presets are defined."""
        # Check key categories are present
        assert "A" in MAXWELL_UNIT_PRESETS
        assert "Hz" in MAXWELL_UNIT_PRESETS
        assert "mm" in MAXWELL_UNIT_PRESETS
        assert "km_per_hour" in MAXWELL_UNIT_PRESETS
        assert "V" in MAXWELL_UNIT_PRESETS
        assert "Ohm" in MAXWELL_UNIT_PRESETS
        assert "N" in MAXWELL_UNIT_PRESETS
        assert "Nm" in MAXWELL_UNIT_PRESETS

    def test_config_load_with_preset_unit(self, tmp_path):
        """Config with preset unit names should resolve correctly."""
        cfg = _valid_config_dict()
        cfg["variables"][0]["unit"] = "millimeter"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(cfg))

        result = load_config(str(config_path))
        assert result.variables[0].resolved_unit() == "mm"
