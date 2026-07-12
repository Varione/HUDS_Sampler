"""Validation script for incremental three-set splitting."""

import sys
sys.path.insert(0, r"E:\大型数据库构建\HUDS_Sampler_GitHub")

from pathlib import Path
import tempfile
import json


def test_split_config():
    """Test SplitConfig is properly defined and validated."""
    from huds_app.core.config import AppConfig, SplitConfig, validate_config

    try:
        config = AppConfig()
        assert hasattr(config, "split"), "AppConfig missing split attribute"
        assert isinstance(config.split, SplitConfig), "split should be SplitConfig"
        assert config.split.train_split == 0.8
        assert config.split.val_split == 0.1
        assert config.split.test_split == 0.1

        # Validation should pass for valid splits
        from huds_app.core.config import VariableConfig, ModelConfig
        config.variables = [VariableConfig(name="x", min=0, max=10, sample_points=5)]
        config.model = ModelConfig(output_names=["y"])
        validate_config(config)

        # Validation should fail for invalid splits
        bad_config = AppConfig()
        bad_config.variables = [VariableConfig(name="x", min=0, max=10, sample_points=5)]
        bad_config.model = ModelConfig(output_names=["y"])
        bad_config.split = SplitConfig(train_split=0.5, val_split=0.3, test_split=0.3)
        try:
            validate_config(bad_config)
            print("[FAIL] Config validation did not reject invalid split ratios")
            return False
        except ValueError:
            pass

        print("[PASS] SplitConfig validation works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] SplitConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_candidate_pool_no_split():
    """Test candidate pool no longer has split column."""
    from huds_app.core.config import AppConfig, VariableConfig, ModelConfig
    from huds_app.data.pool import create_candidate_pool

    try:
        config = AppConfig()
        config.variables = [VariableConfig(name="x", min=0, max=10, sample_points=3)]
        config.model = ModelConfig(output_names=["y"])
        pool_df = create_candidate_pool(config)

        assert "split" not in pool_df.columns, "Pool should not have split column"
        assert "sample_id" in pool_df.columns
        assert "status" in pool_df.columns
        print("[PASS] Candidate pool has no split column")
        return True
    except Exception as e:
        print(f"[FAIL] Candidate pool test failed: {e}")
        return False


def test_split_selected_ids():
    """Test _split_selected_ids produces correct proportions."""
    import numpy as np
    from huds_app.core.config import AppConfig, SplitConfig
    from huds_app.sampling.huds import _split_selected_ids

    try:
        config = AppConfig()
        config.random_seed = 42
        ids = [f"id_{i}" for i in range(100)]
        result = _split_selected_ids(ids, config)

        assert len(result["train"]) == 80, f"Expected 80 train, got {len(result['train'])}"
        assert len(result["val"]) == 10, f"Expected 10 val, got {len(result['val'])}"
        assert len(result["test"]) == 10, f"Expected 10 test, got {len(result['test'])}"

        # Check no duplicates
        all_ids = result["train"] + result["val"] + result["test"]
        assert len(all_ids) == len(set(all_ids)), "Split IDs should be unique"
        assert sorted(all_ids) == sorted(ids), "Split should contain all original IDs"

        print("[PASS] Split selected IDs works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Split selected IDs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_small_batch():
    """Test split handles small batches (ensures train gets at least 1)."""
    from huds_app.core.config import AppConfig
    from huds_app.sampling.huds import _split_selected_ids

    try:
        config = AppConfig()
        config.random_seed = 42
        ids = ["id_0", "id_1"]
        result = _split_selected_ids(ids, config)

        assert len(result["train"]) >= 1, "Train should have at least 1 sample"
        total = len(result["train"]) + len(result["val"]) + len(result["test"])
        assert total == 2, f"All IDs should be split, got {total}"

        print("[PASS] Small batch split works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Small batch split test failed: {e}")
        return False


def test_import_labels_routing():
    """Test import_labels routes samples to correct split files."""
    from huds_app.core.config import AppConfig, VariableConfig, ModelConfig
    from huds_app.data.pool import create_candidate_pool, save_pool_files
    from huds_app.core.storage import RunState, write_csv
    from huds_app.data.validation import import_labels

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            (run_path / "requests").mkdir()
            (run_path / "datasets").mkdir()

            # Create config
            config = AppConfig()
            config.variables = [VariableConfig(name="x", min=0, max=10, sample_points=5)]
            config.model = ModelConfig(output_names=["y"])
            config.split.train_split = 0.8
            config.split.val_split = 0.1
            config.split.test_split = 0.1
            (run_path / "config.json").write_text(json.dumps({
                "variables": [{"name": "x", "min": 0, "max": 10, "sample_points": 5}],
                "model": {"output_names": ["y"]},
                "split": {"train_split": 0.8, "val_split": 0.1, "test_split": 0.1},
            }))

            # Create state.json
            state = RunState(run_dir=str(run_path))
            state.save()

            # Create request with split assignment
            request_df = pd.DataFrame({
                "sample_id": list(range(10)),
                "x": [i * 1.0 for i in range(10)],
                "split_assignment": ["train"] * 8 + ["val"] * 1 + ["test"] * 1,
            })
            write_csv(request_df, run_path / "requests" / "train_step_001_request.csv")

            # Create simulator output
            sim_df = pd.DataFrame({
                "sample_id": list(range(10)),
                "x": [i * 1.0 for i in range(10)],
                "y": [i * 2.0 for i in range(10)],
            })
            sim_path = run_path / "sim_output.csv"
            write_csv(sim_df, sim_path)

            # Import labels
            n = import_labels(str(run_path), step=1, input_path=str(sim_path))
            assert n == 10, f"Should import 10 rows, got {n}"

            # Check split files
            train_labeled = pd.read_csv(run_path / "datasets" / "train_labeled.csv")
            val_labeled = pd.read_csv(run_path / "datasets" / "val_labeled.csv")
            test_labeled = pd.read_csv(run_path / "datasets" / "test_labeled.csv")

            assert len(train_labeled) == 8, f"Expected 8 train, got {len(train_labeled)}"
            assert len(val_labeled) == 1, f"Expected 1 val, got {len(val_labeled)}"
            assert len(test_labeled) == 1, f"Expected 1 test, got {len(test_labeled)}"

            print("[PASS] Import labels routing works correctly")
            return True
    except Exception as e:
        print(f"[FAIL] Import labels routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import pandas as pd

    tests = [
        test_split_config,
        test_candidate_pool_no_split,
        test_split_selected_ids,
        test_split_small_batch,
        test_import_labels_routing,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Validation results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("All improvements validated successfully!")
        sys.exit(0)
    else:
        print("Some tests failed. Please review.")
        sys.exit(1)
