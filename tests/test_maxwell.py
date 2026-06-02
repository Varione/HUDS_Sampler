from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest

from huds_app.config import (
    AppConfig,
    CandidatePoolConfig,
    HUDSConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
    VariableConfig,
)
from huds_app.maxwell import export_maxwell_table, parse_unit_overrides
from huds_app.storage import read_csv, write_csv


@pytest.fixture
def maxwell_config():
    return AppConfig(
        project_name="maxwell_test",
        random_seed=42,
        variables=[
            VariableConfig(name="gap", min=5, max=40, sample_points=13, unit="mm"),
            VariableConfig(name="v", min=0, max=20, sample_points=21, unit="m_per_sec"),
        ],
        candidate_pool=CandidatePoolConfig(total_samples=10, train_ratio=0.8, validation_ratio=0.2),
        model=ModelConfig(output_names=["force"]),
        validation=ValidationConfig(default_size=2),
        training=TrainingConfig(initial_train_size=2, sample_per_step=1, max_steps=1, epochs_per_step=1, batch_size=1, learning_rate=0.001, patience=1, device="cpu"),
        huds=HUDSConfig(use_faiss=False),
    )


class TestParseUnitOverrides:
    def test_parses_multiple_entries(self):
        result = parse_unit_overrides(["gap=mm", "v=m_per_sec"])
        assert result == {"gap": "mm", "v": "m_per_sec"}

    def test_rejects_invalid_entry(self):
        with pytest.raises(ValueError, match="Expected NAME=UNIT"):
            parse_unit_overrides(["gap"])


class TestExportMaxwellTable:
    def test_exports_with_units_from_config(self, tmp_path, maxwell_config):
        input_df = pd.DataFrame(
            {
                "sample_id": [101, 102, 103],
                "gap": [5, 8, 11],
                "v": [0, 1, 2],
            }
        )
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "ParametricSetup1_Table.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(input_path, output_path, config=maxwell_config)

        output_df = read_csv(output_path)
        assert output_df.columns.tolist() == ["*", "gap", "v"]
        assert output_df.iloc[0].to_dict() == {"*": 1, "gap": "5mm", "v": "0m_per_sec"}
        assert output_df.iloc[2].to_dict() == {"*": 3, "gap": "11mm", "v": "2m_per_sec"}

    def test_exports_with_unit_overrides_without_config(self, tmp_path):
        input_df = pd.DataFrame(
            {
                "sample_id": [1, 2],
                "gap": [0.005, 0.008],
                "v": [0, 3],
            }
        )
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "maxwell.csv"
        write_csv(input_df, input_path)

        export_maxwell_table(
            input_path,
            output_path,
            config=None,
            units={"gap": "mm", "v": "m_per_sec"},
        )

        output_df = read_csv(output_path)
        assert output_df.iloc[0].to_dict() == {"*": 1, "gap": "0.005mm", "v": "0m_per_sec"}

    def test_uses_config_path(self, tmp_path, maxwell_config):
        input_df = pd.DataFrame(
            {
                "sample_id": [1],
                "gap": [14],
                "v": [4],
            }
        )
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "maxwell.csv"
        config_path = tmp_path / "config.json"
        write_csv(input_df, input_path)
        config_path.write_text(json.dumps(asdict(maxwell_config)), encoding="utf-8")

        export_maxwell_table(input_path, output_path, config=config_path)

        output_df = read_csv(output_path)
        assert output_df.iloc[0].to_dict() == {"*": 1, "gap": "14mm", "v": "4m_per_sec"}

    def test_requires_units_for_all_variables(self, tmp_path):
        input_df = pd.DataFrame({"sample_id": [1], "gap": [5]})
        input_path = tmp_path / "request.csv"
        output_path = tmp_path / "maxwell.csv"
        write_csv(input_df, input_path)

        with pytest.raises(ValueError, match="Missing unit for variable 'gap'"):
            export_maxwell_table(input_path, output_path)
