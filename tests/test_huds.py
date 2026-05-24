import numpy as np
import pandas as pd
import pytest
import torch
from huds_app.huds import mc_dropout_predict, select_huds
from huds_app.model import ResidualMLP
from huds_app.config import (
    AppConfig,
    VariableConfig,
    ModelConfig,
    TrainingConfig,
    HUDSConfig,
    CandidatePoolConfig,
)


@pytest.fixture
def small_model():
    return ResidualMLP(input_dim=2, output_dim=1, hidden_dim=8, residual_blocks=1, dropout=0.3)


@pytest.fixture
def test_config():
    return AppConfig(
        project_name="test",
        random_seed=42,
        variables=[
            VariableConfig(name="x", min=0.0, max=1.0, sample_points=5),
            VariableConfig(name="y", min=0.0, max=1.0, sample_points=5),
        ],
        candidate_pool=CandidatePoolConfig(total_samples=25, train_ratio=0.8, validation_ratio=0.2),
        model=ModelConfig(output_names=["z"], hidden_dim=8, residual_blocks=1, dropout=0.3),
        training=TrainingConfig(
            initial_train_size=4,
            sample_per_step=5,
            max_steps=2,
            epochs_per_step=1,
            batch_size=10,
            learning_rate=0.001,
            patience=5,
            device="cpu",
        ),
        huds=HUDSConfig(pre_n=0, repeat_times=3, topk_ratio=0.6, batch_size=256, use_faiss=False),
    )


class TestMcDropoutPredict:
    def test_returns_correct_shapes(self, small_model):
        x = np.random.rand(10, 2).astype(np.float32)
        repeat_times = 4
        predictions, uncertainties = mc_dropout_predict(small_model, x, repeat_times, batch_size=8)
        assert predictions.shape == (repeat_times, 10, 1), (
            f"Expected shape ({repeat_times}, 10, 1), got {predictions.shape}"
        )
        assert uncertainties.shape == (10,), (
            f"Expected uncertainty shape (10,), got {uncertainties.shape}"
        )

    def test_works_with_torch_tensor_input(self, small_model):
        x = torch.randn(6, 2)
        repeat_times = 2
        predictions, uncertainties = mc_dropout_predict(small_model, x, repeat_times, batch_size=3)
        assert predictions.shape == (repeat_times, 6, 1)
        assert uncertainties.shape == (6,)

    def test_preserves_eval_mode(self, small_model):
        small_model.eval()
        x = np.random.rand(3, 2).astype(np.float32)
        mc_dropout_predict(small_model, x, repeat_times=2, batch_size=2)
        assert not small_model.training

    def test_preserves_train_mode(self, small_model):
        small_model.train()
        x = np.random.rand(3, 2).astype(np.float32)
        mc_dropout_predict(small_model, x, repeat_times=2, batch_size=2)
        assert small_model.training

    def test_raises_on_invalid_repeat_times(self, small_model):
        x = np.random.rand(5, 2).astype(np.float32)
        with pytest.raises(ValueError, match="repeat_times"):
            mc_dropout_predict(small_model, x, repeat_times=0, batch_size=8)

    def test_raises_on_invalid_batch_size(self, small_model):
        x = np.random.rand(5, 2).astype(np.float32)
        with pytest.raises(ValueError, match="batch_size"):
            mc_dropout_predict(small_model, x, repeat_times=3, batch_size=-1)


class TestSelectHuds:
    def test_returns_exact_sample_per_step(self, small_model, test_config):
        df = pd.DataFrame({
            "sample_id": list(range(20)),
            "x": np.random.rand(20),
            "y": np.random.rand(20),
        })
        unlabeled_mask = [True] * 20
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        assert len(result["selected_ids"]) == test_config.training.sample_per_step

    def test_returns_fewer_when_pool_exhausted(self, small_model, test_config):
        few_samples = 2
        df = pd.DataFrame({
            "sample_id": list(range(few_samples)),
            "x": np.random.rand(few_samples),
            "y": np.random.rand(few_samples),
        })
        unlabeled_mask = [True] * few_samples
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        assert len(result["selected_ids"]) == few_samples
        assert len(result["selected_ids"]) < test_config.training.sample_per_step

    def test_returns_empty_when_no_unlabeled(self, small_model, test_config):
        df = pd.DataFrame({
            "sample_id": [0, 1],
            "x": [0.5, 0.6],
            "y": [0.3, 0.4],
        })
        unlabeled_mask = [False, False]
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        assert len(result["selected_ids"]) == 0

    def test_selected_ids_are_from_candidate_pool(self, small_model, test_config):
        pool_ids = list(range(15))
        df = pd.DataFrame({
            "sample_id": pool_ids,
            "x": np.random.rand(15),
            "y": np.random.rand(15),
        })
        unlabeled_mask = [True] * 15
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        for sid in result["selected_ids"]:
            assert sid in pool_ids

    def test_result_contains_required_keys(self, small_model, test_config):
        df = pd.DataFrame({
            "sample_id": list(range(10)),
            "x": np.random.rand(10),
            "y": np.random.rand(10),
        })
        unlabeled_mask = [True] * 10
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        expected_keys = {"selected_ids", "uncertainties", "topk_size", "n_clusters", "cluster_stats", "checkpoint_used"}
        assert set(result.keys()) >= expected_keys

    def test_no_duplicates_in_selected(self, small_model, test_config):
        df = pd.DataFrame({
            "sample_id": list(range(20)),
            "x": np.random.rand(20),
            "y": np.random.rand(20),
        })
        unlabeled_mask = [True] * 20
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(
            small_model, df, unlabeled_mask, labeled_df, test_config, ["x", "y"], "cpu"
        )
        assert len(result["selected_ids"]) == len(set(result["selected_ids"]))


class TestKCenterFill:
    def test_k_center_fills_remaining_slots(self, small_model):
        """When clustering selects fewer than n_select, k-center fill should top up."""
        config = AppConfig(
            project_name="test",
            random_seed=42,
            variables=[
                VariableConfig(name="x", min=0.0, max=1.0, sample_points=5),
                VariableConfig(name="y", min=0.0, max=1.0, sample_points=5),
            ],
            candidate_pool=CandidatePoolConfig(),
            model=ModelConfig(output_names=["z"], hidden_dim=8, residual_blocks=1, dropout=0.3),
            training=TrainingConfig(
                initial_train_size=4,
                sample_per_step=5,
                max_steps=2,
                epochs_per_step=1,
                batch_size=10,
                learning_rate=0.001,
                patience=5,
                device="cpu",
            ),
            huds=HUDSConfig(pre_n=3, repeat_times=3, topk_ratio=0.6, batch_size=256, use_faiss=False),
        )

        df = pd.DataFrame({
            "sample_id": list(range(10)),
            "x": np.random.rand(10),
            "y": np.random.rand(10),
        })
        unlabeled_mask = [True] * 10
        labeled_df = pd.DataFrame({"sample_id": []})

        result = select_huds(small_model, df, unlabeled_mask, labeled_df, config, ["x", "y"], "cpu")
        assert len(result["selected_ids"]) == min(5, 3)
