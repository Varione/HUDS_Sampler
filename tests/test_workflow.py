from __future__ import annotations

import json
from dataclasses import asdict

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
from huds_app.sampling.huds import run_huds_sampling
from huds_app.data.pool import create_candidate_pool, save_pool_files, split_pool
from huds_app.core.storage import RunState, read_csv, write_csv
from huds_app.model.train import train_model
from huds_app.data.validation import export_initial_train_request, export_validation_request, import_labels
from huds_app.interface.workflow import evaluate, predict


@pytest.fixture
def workflow_config():
    return AppConfig(
        project_name="workflow_test",
        random_seed=7,
        variables=[
            VariableConfig(name="x1", min=0.0, max=1.0, sample_points=5),
            VariableConfig(name="x2", min=0.0, max=1.0, sample_points=5),
        ],
        candidate_pool=CandidatePoolConfig(total_samples=120, train_ratio=0.8, validation_ratio=0.2),
        model=ModelConfig(output_names=["y1", "y2"], hidden_dim=16, encoder_blocks=1, dropout=0.2),
        validation=ValidationConfig(default_size=20),
        training=TrainingConfig(
            initial_train_size=24,
            sample_per_step=8,
            max_steps=2,
            epochs_per_step=5,
            batch_size=16,
            learning_rate=0.01,
            patience=3,
            device="cpu",
        ),
        huds=HUDSConfig(repeat_times=3, topk_ratio=0.6, batch_size=32, use_faiss=False),
    )


@pytest.fixture
def trained_run(tmp_path, workflow_config):
    run_dir = tmp_path / "run"

    pool_df = create_candidate_pool(workflow_config)
    train_df, valid_df = split_pool(pool_df, workflow_config, workflow_config.random_seed)
    save_pool_files(pool_df, train_df, valid_df, run_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(workflow_config), file)

    RunState(run_dir=str(run_dir)).save()

    export_validation_request(run_dir, workflow_config, size=20)
    validation_request = read_csv(run_dir / "requests" / "validation_request.csv")
    for output_name in workflow_config.model.output_names:
        validation_request[output_name] = (
            validation_request["x1"].to_numpy() * 0.4
            + validation_request["x2"].to_numpy() * 0.2
        )
    validation_path = run_dir / "validation_labels.csv"
    validation_request.to_csv(validation_path, index=False)
    import_labels(run_dir, kind="validation", step=None, input_path=validation_path)

    export_initial_train_request(run_dir, workflow_config)
    train_request = read_csv(run_dir / "requests" / "train_step_000_request.csv")
    for output_name in workflow_config.model.output_names:
        train_request[output_name] = (
            train_request["x1"].to_numpy() * 0.7
            - train_request["x2"].to_numpy() * 0.1
        )
    train_path = run_dir / "train_labels.csv"
    train_request.to_csv(train_path, index=False)
    import_labels(run_dir, kind="train", step=0, input_path=train_path)

    train_model(run_dir, workflow_config)
    return run_dir


class TestWorkflowInference:
    def test_predict_writes_expected_output_columns(self, trained_run):
        input_df = pd.DataFrame(
            {
                "sample_id": [9001, 9002],
                "x1": [0.2, 0.8],
                "x2": [0.3, 0.1],
            }
        )
        input_path = trained_run / "predict_input.csv"
        output_path = trained_run / "predict_output.csv"
        write_csv(input_df, input_path)

        predict(trained_run, input_path, output_path)

        predicted_df = read_csv(output_path)
        assert list(predicted_df.columns) == ["sample_id", "x1", "x2", "y1", "y2"]
        assert len(predicted_df) == 2

    def test_evaluate_returns_metrics_for_each_output(self, trained_run):
        metrics = evaluate(trained_run)

        assert "r2_avg" in metrics
        assert "r2_y1" in metrics
        assert "r2_y2" in metrics
        assert "rmse_y1" in metrics
        assert "rmse_y2" in metrics


class TestSamplingStepValidation:
    def test_rejects_duplicate_sampling_step(self, trained_run, workflow_config):
        run_huds_sampling(trained_run, workflow_config, step=1)

        with pytest.raises(ValueError, match="already exists"):
            run_huds_sampling(trained_run, workflow_config, step=1)

    def test_rejects_skipping_ahead(self, trained_run, workflow_config):
        with pytest.raises(ValueError, match="next step in sequence"):
            run_huds_sampling(trained_run, workflow_config, step=2)

    def test_rejects_sampling_beyond_max_steps(self, trained_run, workflow_config):
        run_huds_sampling(trained_run, workflow_config, step=1)
        request_df = read_csv(trained_run / "requests" / "train_step_001_request.csv")
        for output_name in workflow_config.model.output_names:
            request_df[output_name] = np.zeros(len(request_df))
        labels_path = trained_run / "train_step_001_labels.csv"
        request_df.to_csv(labels_path, index=False)
        import_labels(trained_run, kind="train", step=1, input_path=labels_path)
        train_model(trained_run, workflow_config)

        with pytest.raises(ValueError, match="exceeds configured max_steps"):
            run_huds_sampling(trained_run, workflow_config, step=3)
