"""Tests for LHS sampling and candidate pool splitting."""

import pytest
import numpy as np
import pandas as pd

from huds_app.config import AppConfig, VariableConfig, CandidatePoolConfig
from huds_app.sampling import create_candidate_pool


@pytest.fixture
def config():
    """Basic single-variable config with 100 samples."""
    return AppConfig(
        random_seed=42,
        variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
        candidate_pool=CandidatePoolConfig(total_samples=100, train_ratio=0.8, validation_ratio=0.2),
    )


@pytest.fixture
def pool(config):
    """Create a candidate pool from config."""
    return create_candidate_pool(config)


class TestCandidatePoolShapeAndColumns:
    def test_returns_correct_shape(self, pool, config):
        assert pool.shape == (config.candidate_pool.total_samples, len(config.variables) + 3)

    def test_columns_include_sample_id_split_status_and_variables(self, pool):
        expected = {"sample_id", "split", "status"} | {v.name for v in [VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)]}
        assert set(pool.columns) == expected

    def test_sample_id_is_sequential(self, pool):
        assert (pool["sample_id"] == np.arange(100)).all()


class TestValueBounds:
    def test_values_within_bounds(self, pool):
        x = pool["x"]
        assert (x >= 0.0).all() and (x <= 1.0).all()

    def test_snap_to_levels_produces_discrete_values(self, config):
        snapped_pool = create_candidate_pool(config, snap_to_levels=True)
        levels = np.linspace(0.0, 1.0, 5)
        for v in snapped_pool["x"]:
            assert any(np.isclose(v, level) for level in levels), f"{v} not close to any level"

    def test_snap_to_levels_reduces_unique_values(self, config):
        normal = create_candidate_pool(config, snap_to_levels=False)
        snapped = create_candidate_pool(config, snap_to_levels=True)
        assert snapped["x"].nunique() <= 5
        assert normal["x"].nunique() > 5


class TestDeterminism:
    def test_same_seed_gives_same_result(self, config):
        pool_a = create_candidate_pool(config)
        pool_b = create_candidate_pool(config)
        pd.testing.assert_frame_equal(pool_a, pool_b)

    def test_different_seed_gives_different_result(self):
        cfg_a = AppConfig(
            random_seed=42,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            candidate_pool=CandidatePoolConfig(total_samples=100),
        )
        cfg_b = AppConfig(
            random_seed=99,
            variables=[VariableConfig(name="x", min=0.0, max=1.0, sample_points=5)],
            candidate_pool=CandidatePoolConfig(total_samples=100),
        )
        pool_a = create_candidate_pool(cfg_a)
        pool_b = create_candidate_pool(cfg_b)
        assert not np.allclose(pool_a["x"].values, pool_b["x"].values)


class TestSplitRatios:
    def test_train_count_matches_ratio(self, pool, config):
        train_count = (pool["split"] == "train_pool").sum()
        expected = int(config.candidate_pool.total_samples * config.candidate_pool.train_ratio)
        assert train_count == expected

    def test_validation_count_matches_ratio(self, pool, config):
        valid_count = (pool["split"] == "validation_pool").sum()
        expected = int(
            config.candidate_pool.total_samples
            - config.candidate_pool.total_samples * config.candidate_pool.train_ratio
        )
        assert valid_count == expected

    def test_all_rows_have_split(self, pool):
        assert set(pool["split"].unique()).issubset({"train_pool", "validation_pool"})
