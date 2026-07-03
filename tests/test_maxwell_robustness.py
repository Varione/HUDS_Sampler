"""Robustness tests for Maxwell parametric table export and round-trip import.

Covers edge cases identified from the complete workflow analysis:
- NaN / Inf / -Inf value handling in export_maxwell_table
- Large integer sample_id precision (>2^53)
- Negative zero normalization
- Scientific notation values
- Sample_id round-trip through Maxwell boundary
- CSV BOM encoding handling
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from huds_app.core.config import (
    AppConfig,
    CandidatePoolConfig,
    HUDSConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
    VariableConfig,
)
from huds_app.data.schema import SAMPLE_ID_COLUMN
from huds_app.interface.maxwell import export_maxwell_table, parse_unit_overrides
from huds_app.core.storage import read_csv, write_csv


@pytest.fixture
def maxwell_config():
    return AppConfig(
        project_name="maxwell_robust",
        random_seed=42,
        variables=[
            VariableConfig(name="i1", min=0, max=100, sample_points=10, unit="A"),
            VariableConfig(name="f", min=1, max=50, sample_points=10, unit="Hz"),
            VariableConfig(name="vel", min=0, max=30, sample_points=10, unit="km_per_hour"),
        ],
        candidate_pool=CandidatePoolConfig(total_samples=20, train_ratio=0.8, validation_ratio=0.2),
        model=ModelConfig(output_names=["force", "torque"]),
        validation=ValidationConfig(default_size=5),
        training=TrainingConfig(
            initial_train_size=5, sample_per_step=3, max_steps=2,
            epochs_per_step=1, batch_size=1, learning_rate=0.001, patience=1, device="cpu",
        ),
        huds=HUDSConfig(use_faiss=False),
    )


# =============================================================================
# 1. NaN / Inf / -Inf edge cases in _format_maxwell_value
# =============================================================================

class TestMaxwellNaNInfHandling:
    """Test that NaN and Inf values are caught before Maxwell export."""

    def test_nan_raises_value_error(self, tmp_path, maxwell_config):
        """NaN in variable column must raise ValueError."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [float("nan")],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Cannot format non-numeric value"):
            export_maxwell_table(input_path, output_path, config=maxwell_config)

    def test_positive_inf_raises_value_error(self, tmp_path, maxwell_config):
        """Positive Inf must raise ValueError."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [float("inf")],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Cannot format non-numeric value"):
            export_maxwell_table(input_path, output_path, config=maxwell_config)

    def test_negative_inf_raises_value_error(self, tmp_path, maxwell_config):
        """Negative Inf must raise ValueError."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [float("-inf")],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Cannot format non-numeric value"):
            export_maxwell_table(input_path, output_path, config=maxwell_config)

    def test_numpy_nan_raises_value_error(self, tmp_path, maxwell_config):
        """np.nan must raise ValueError."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [np.nan],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Cannot format non-numeric value"):
            export_maxwell_table(input_path, output_path, config=maxwell_config)

    def test_numpy_inf_raises_value_error(self, tmp_path, maxwell_config):
        """np.inf must raise ValueError."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [np.inf],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Cannot format non-numeric value"):
            export_maxwell_table(input_path, output_path, config=maxwell_config)


# =============================================================================
# 2. Negative zero and precision edge cases
# =============================================================================

class TestMaxwellValueFormatting:
    """Test _format_maxwell_value handles numeric edge cases correctly."""

    def test_negative_zero_becomes_zero(self, tmp_path, maxwell_config):
        """-0.0 must be formatted as '0A', not '-0A'."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [-0.0],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.loc[0, "i1"] == "0A"

    def test_very_small_value_preserved(self, tmp_path, maxwell_config):
        """Very small values should be formatted correctly."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [0.000001],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.loc[0, "i1"] == "1E-6A" or result.loc[0, "i1"] == "0.000001A"

    def test_very_large_value_formatted(self, tmp_path, maxwell_config):
        """Very large values should be formatted as full integer."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [1e12],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.loc[0, "i1"] == "1000000000000A"

    def test_integer_values_no_trailing_zeros(self, tmp_path, maxwell_config):
        """Integer-like floats should not have decimal points."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [18.0],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.loc[0, "i1"] == "18A"
        assert result.loc[0, "f"] == "5Hz"
        assert result.loc[0, "vel"] == "1km_per_hour"

    def test_decimal_precision_preserved(self, tmp_path, maxwell_config):
        """Decimal values should preserve meaningful precision."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [18.5],
            "f": [3.14159],
            "vel": [21.75],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.loc[0, "i1"] == "18.5A"
        assert result.loc[0, "vel"] == "21.75km_per_hour"


# =============================================================================
# 3. Large sample_id precision (>2^53)
# =============================================================================

class TestMaxwellLargeSampleIds:
    """Test that large sample_id values survive the export round-trip."""

    def test_sample_id_gt_2pow53_in_request(self, tmp_path, maxwell_config):
        """sample_id > 2^53 must be preserved in request CSV."""
        large_id = 2**53 + 7  # 9007199254740991 + 7
        input_df = pd.DataFrame({
            "sample_id": [large_id],
            "i1": [18.0],
            "f": [5.0],
            "vel": [1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        # Verify request CSV preserves the ID
        req_read = read_csv(input_path)
        assert int(req_read.iloc[0]["sample_id"]) == large_id

        # Export to Maxwell format (sample_id is dropped, replaced by *)
        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        # Maxwell output should have * column, not sample_id
        assert "*" in result.columns
        assert "sample_id" not in result.columns
        assert result.iloc[0]["*"] == 1

    def test_multiple_large_sample_ids(self, tmp_path, maxwell_config):
        """Multiple large sample_ids must all be preserved."""
        ids = [2**53 + i for i in range(5)]
        input_df = pd.DataFrame({
            "sample_id": ids,
            "i1": [18.0, 33.0, 94.0, 12.0, 55.0],
            "f": [5.0, 6.0, 3.0, 16.0, 2.0],
            "vel": [1.0, 3.0, 1.0, 21.0, 4.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        req_read = read_csv(input_path)
        for i, sid in enumerate(ids):
            assert int(req_read.iloc[i]["sample_id"]) == sid


# =============================================================================
# 4. Maxwell round-trip: request → export → simulator output → import
# =============================================================================

class TestMaxwellRoundTrip:
    """Test the complete Maxwell boundary round-trip."""

    def test_maxwell_export_preserves_row_order(self, tmp_path, maxwell_config):
        """Row order in request must match * index in Maxwell output."""
        input_df = pd.DataFrame({
            "sample_id": [101, 202, 303, 404],
            "i1": [18.0, 33.0, 94.0, 12.0],
            "f": [5.0, 6.0, 3.0, 16.0],
            "vel": [1.0, 3.0, 1.0, 21.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "maxwell.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        # Verify * column is sequential
        assert result["*"].tolist() == [1, 2, 3, 4]

        # Verify values match original order
        assert result.iloc[0]["i1"] == "18A"
        assert result.iloc[3]["i1"] == "12A"

    def test_maxwell_export_drops_sample_id(self, tmp_path, maxwell_config):
        """sample_id column must NOT appear in Maxwell output."""
        input_df = pd.DataFrame({
            "sample_id": [1, 2],
            "i1": [18.0, 33.0],
            "f": [5.0, 6.0],
            "vel": [1.0, 3.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "maxwell.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert list(result.columns) == ["*", "i1", "f", "vel"]

    def test_simulator_output_must_have_sample_id(self, tmp_path, maxwell_config):
        """Simulator output must contain sample_id to be importable."""
        # Simulate Maxwell output WITHOUT sample_id (common mistake)
        sim_df = pd.DataFrame({
            "*": [1, 2],
            "i1": [18.0, 33.0],
            "f": [5.0, 6.0],
            "vel": [1.0, 3.0],
            "force": [100.5, 200.3],
            "torque": [50.1, 75.2],
        })
        sim_path = tmp_path / "sim_output.csv"
        write_csv(sim_df, sim_path)

        # This should fail validation because sample_id is missing
        from huds_app.data.validation import validate_simulator_output

        errors = validate_simulator_output(sim_df, maxwell_config)
        assert any("sample_id" in e for e in errors), f"Expected sample_id error, got: {errors}"

    def test_simulator_output_with_sample_id_passes(self, tmp_path, maxwell_config):
        """Simulator output WITH sample_id must pass validation."""
        sim_df = pd.DataFrame({
            "sample_id": [101, 202],
            "i1": [18.0, 33.0],
            "f": [5.0, 6.0],
            "vel": [1.0, 3.0],
            "force": [100.5, 200.3],
            "torque": [50.1, 75.2],
        })

        from huds_app.data.validation import validate_simulator_output

        errors = validate_simulator_output(sim_df, maxwell_config)
        assert len(errors) == 0, f"Expected no errors, got: {errors}"


# =============================================================================
# 5. CSV BOM encoding robustness
# =============================================================================

class TestCsvBomHandling:
    """Test that CSV files with UTF-8 BOM can be read correctly."""

    def test_read_csv_with_bom(self, tmp_path):
        """CSV with UTF-8 BOM should be readable."""
        df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "i1": [18.0, 33.0, 94.0],
            "f": [5.0, 6.0, 3.0],
        })
        path = tmp_path / "bom.csv"

        # Write with BOM manually
        with open(path, "w", encoding="utf-8-sig") as f:
            df.to_csv(f, index=False)

        # Should read without error
        result = read_csv(path)
        assert len(result) == 3

        # Column names should not have BOM prefix
        assert "sample_id" in result.columns
        assert "\ufeffsample_id" not in result.columns

    def test_write_read_roundtrip(self, tmp_path):
        """Write then read CSV should preserve data."""
        original = pd.DataFrame({
            "sample_id": [1, 2],
            "i1": [18.0, 33.0],
        })
        path = tmp_path / "roundtrip.csv"

        write_csv(original, path)
        result = read_csv(path)

        assert list(result.columns) == list(original.columns)
        assert len(result) == len(original)


# =============================================================================
# 6. Maxwell variable.csv format matching (actual file format)
# =============================================================================

class TestMaxwellFormatMatching:
    """Verify export matches the actual Maxwell parametric table format."""

    def test_matches_maxwell_variable_csv_format(self, tmp_path, maxwell_config):
        """Export must match the format in maxwell_variable.csv."""
        # Exact data from maxwell_variable.csv reference
        input_df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "i1": [18.0, 18.0, 33.0, 94.0],
            "f": [5.0, 16.0, 6.0, 3.0],
            "vel": [1.0, 21.0, 3.0, 1.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "ParametricSetup1_Table.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        # Verify format matches reference: *,i1,f,vel
        assert list(result.columns) == ["*", "i1", "f", "vel"]

        # Verify values match expected format (number + unit, no space)
        assert result.iloc[0]["*"] == 1
        assert result.iloc[0]["i1"] == "18A"
        assert result.iloc[0]["f"] == "5Hz"
        assert result.iloc[0]["vel"] == "1km_per_hour"

        assert result.iloc[1]["i1"] == "18A"
        assert result.iloc[1]["f"] == "16Hz"
        assert result.iloc[1]["vel"] == "21km_per_hour"

    def test_unit_with_underscore_preserved(self, tmp_path, maxwell_config):
        """Units with underscores (km_per_hour) must be preserved exactly."""
        input_df = pd.DataFrame({
            "sample_id": [1],
            "i1": [18.0],
            "f": [5.0],
            "vel": [21.0],
        })
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "output.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)
        result = read_csv(output_path)

        assert result.iloc[0]["vel"] == "21km_per_hour"


# =============================================================================
# 7. Unit override robustness
# =============================================================================

class TestUnitOverrideRobustness:
    """Test unit override parsing and application edge cases."""

    def test_unit_override_with_spaces(self):
        """Unit overrides with extra whitespace should be trimmed."""
        result = parse_unit_overrides(["gap = mm", "  v  =  m_per_sec  "])
        assert result == {"gap": "mm", "v": "m_per_sec"}

    def test_unit_override_empty_name_raises(self):
        """Empty name in unit override must raise."""
        with pytest.raises(ValueError, match="Invalid unit override"):
            parse_unit_overrides(["=mm"])

    def test_unit_override_empty_value_raises(self):
        """Empty value in unit override must raise."""
        with pytest.raises(ValueError, match="Invalid unit override"):
            parse_unit_overrides(["gap="])

    def test_unit_override_none_input(self):
        """None input should return empty dict."""
        assert parse_unit_overrides(None) == {}

    def test_unit_override_empty_list(self):
        """Empty list should return empty dict."""
        assert parse_unit_overrides([]) == {}
