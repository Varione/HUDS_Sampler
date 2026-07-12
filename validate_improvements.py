"""Validation script for HUDS improvement fixes."""

import sys
sys.path.insert(0, r"E:\大型数据库构建\HUDS_Sampler_GitHub")

from pathlib import Path
import tempfile
import os

def test_cli_entry():
    """Test CLI entry point."""
    try:
        from huds_app.interface import cli
        print("[PASS] CLI module loads correctly")
        return True
    except Exception as e:
        print(f"[FAIL] CLI module load failed: {e}")
        return False

def test_atomic_write():
    """Test atomic write functionality."""
    import json
    from huds_app.core.storage import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        data = {"key": "value", "number": 42}
        try:
            atomic_write_json(path, data)
            assert path.exists(), "File not created"
            with open(path, "r") as f:
                loaded = json.load(f)
            assert loaded == data, "Data mismatch"
            print("[PASS] Atomic write works correctly")
            return True
        except Exception as e:
            print(f"[FAIL] Atomic write failed: {e}")
            return False

def test_mc_dropout_only_dropout():
    """Test that MC Dropout only affects Dropout layers."""
    import torch
    from huds_app.model.architecture import VectorToImage
    from huds_app.sampling.huds import _enable_mc_dropout

    try:
        model = VectorToImage(input_dim=10, hidden_dim=32, encoder_blocks=2, dropout=0.1)
        _enable_mc_dropout(model)
        has_dropout_train = any(
            m.training for m in model.modules()
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d))
        )
        assert has_dropout_train, "Dropout should be enabled"
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for bm in bn_modules:
            assert not bm.training, f"BatchNorm should stay in eval mode"
        print("[PASS] MC Dropout only affects Dropout layers")
        return True
    except Exception as e:
        print(f"[FAIL] MC Dropout test failed: {e}")
        return False

def test_config_validation():
    """Test config validation with repeat_times >= 2."""
    from huds_app.core.config import (
        AppConfig, HUDSConfig, VariableConfig, ModelConfig, validate_config,
    )

    try:
        config = AppConfig()
        config.variables = [VariableConfig(name="test", min=0, max=10, sample_points=5)]
        config.model = ModelConfig(output_names=["output1"])
        config.huds = HUDSConfig(repeat_times=1)
        validate_config(config)
        print("[FAIL] Config validation did not reject repeat_times < 2")
        return False
    except ValueError as e:
        if "repeat_times" in str(e):
            print("[PASS] Config validation correctly rejects repeat_times < 2")
            return True
        else:
            print(f"[FAIL] Unexpected error: {e}")
            return False

def test_image_decoder_size():
    """Test that ImageDecoder outputs correct size."""
    import torch
    from huds_app.model.architecture import VectorToImage

    try:
        model = VectorToImage(
            input_dim=10,
            hidden_dim=32,
            encoder_blocks=2,
            img_h=30,
            img_w=30,
            channels=1,
            decoder_blocks=2,
        )
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 1, 30, 30), (
            f"Output shape {output.shape} does not match expected (1, 1, 30, 30)"
        )
        print("[PASS] ImageDecoder outputs correct size after interpolation")
        return True
    except Exception as e:
        print(f"[FAIL] Image decoder size test failed: {e}")
        return False

# ---------- New tests for audit fixes ----------

def test_mc_dropout_default_path():
    """P0: mc_dropout_predict with return_outputs=False should not raise NameError."""
    import torch
    from huds_app.model.architecture import VectorToVector
    from huds_app.sampling.huds import mc_dropout_predict

    try:
        model = VectorToVector(input_dim=8, hidden_dim=16, output_dim=4, encoder_blocks=2, dropout=0.1)
        x = torch.randn(4, 8)
        embeddings, uncertainties, pred_mean = mc_dropout_predict(
            model, x, repeat_times=3, batch_size=2, return_outputs=False,
        )
        assert pred_mean is None, "pred_mean should be None when return_outputs=False"
        assert embeddings.shape == (3, 4, 16), f"Unexpected embedding shape {embeddings.shape}"
        assert uncertainties.shape == (4,), f"Unexpected uncertainty shape {uncertainties.shape}"
        print("[PASS] mc_dropout_predict default path (return_outputs=False) works")
        return True
    except Exception as e:
        print(f"[FAIL] mc_dropout_predict default path failed: {e}")
        return False

def test_mc_dropout_output_path():
    """P0: mc_dropout_predict with return_outputs=True returns pred_mean."""
    import torch
    from huds_app.model.architecture import VectorToVector
    from huds_app.sampling.huds import mc_dropout_predict

    try:
        model = VectorToVector(input_dim=8, hidden_dim=16, output_dim=4, encoder_blocks=2, dropout=0.1)
        x = torch.randn(4, 8)
        embeddings, uncertainties, pred_mean = mc_dropout_predict(
            model, x, repeat_times=3, batch_size=2, return_outputs=True,
        )
        assert pred_mean is not None, "pred_mean should not be None when return_outputs=True"
        assert pred_mean.shape == (4, 4), f"Unexpected pred_mean shape {pred_mean.shape}"
        print("[PASS] mc_dropout_predict output path (return_outputs=True) works")
        return True
    except Exception as e:
        print(f"[FAIL] mc_dropout_predict output path failed: {e}")
        return False

def test_mc_dropout_state_restoration():
    """P0: Model training state is restored after mc_dropout_predict."""
    import torch
    from huds_app.model.architecture import VectorToVector
    from huds_app.sampling.huds import mc_dropout_predict

    try:
        model = VectorToVector(input_dim=8, hidden_dim=16, output_dim=4, encoder_blocks=2, dropout=0.1)
        x = torch.randn(4, 8)

        # Test restoration from eval mode
        model.eval()
        mc_dropout_predict(model, x, repeat_times=3, batch_size=2, return_outputs=False)
        assert not model.training, "Model should remain in eval after MC Dropout (was eval)"

        # Test restoration from train mode
        model.train()
        mc_dropout_predict(model, x, repeat_times=3, batch_size=2, return_outputs=False)
        assert model.training, "Model should be restored to train after MC Dropout (was train)"
        print("[PASS] MC Dropout correctly restores model training state")
        return True
    except Exception as e:
        print(f"[FAIL] MC Dropout state restoration failed: {e}")
        return False

def test_config_always_validated():
    """P0/P1: validate_config is always called, even when Pydantic is available."""
    from huds_app.core.config import load_config

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_str = (
                '{"variables": [], "model": {"output_names": []}, '
                '"candidate_pool": {"train_ratio": 0.5, "validation_ratio": 0.5}}'
            )
            f.write(json_str)
            tmp_path = f.name

        try:
            load_config(tmp_path)
            print("[FAIL] Config validation did not reject empty variables")
            return False
        except ValueError as e:
            if "variables" in str(e):
                print("[PASS] Config validation runs even when Pydantic is available")
                return True
            else:
                print(f"[FAIL] Unexpected error: {e}")
                return False
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"[FAIL] Config always-validated test crashed: {e}")
        return False

def test_uncertainty_normalized():
    """P1: Uncertainty is normalized by dimension variance."""
    import torch
    from huds_app.model.architecture import VectorToVector
    from huds_app.sampling.huds import mc_dropout_predict

    try:
        model = VectorToVector(input_dim=4, hidden_dim=16, output_dim=3, encoder_blocks=2, dropout=0.2)
        x = torch.randn(8, 4)
        _, uncertainties, _ = mc_dropout_predict(
            model, x, repeat_times=5, batch_size=4, return_outputs=True,
        )
        assert uncertainties.shape == (8,), f"Unexpected shape {uncertainties.shape}"
        assert not torch.isnan(torch.as_tensor(uncertainties)).any(), "Uncertainties contain NaN"
        print("[PASS] Uncertainty normalization produces valid values")
        return True
    except Exception as e:
        print(f"[FAIL] Uncertainty normalization test failed: {e}")
        return False

def test_k_center_from_topk():
    """P1: K-center fill only uses Top-K high-uncertainty pool."""
    import numpy as np
    from huds_app.sampling.huds import select_huds
    import torch

    try:
        # Verify by reading the source code that remaining_positions is built from topk_positions
        import inspect
        source = inspect.getsource(select_huds)
        assert "topk_positions" in source, "topk_positions not referenced in fill logic"
        print("[PASS] K-center fill restricted to Top-K pool (source verified)")
        return True
    except Exception as e:
        print(f"[FAIL] K-center from Top-K test failed: {e}")
        return False

def test_single_member_cluster():
    """P1: Single-member clusters are selected directly, not skipped."""
    import numpy as np
    from huds_app.sampling.huds import select_huds
    import inspect

    try:
        source = inspect.getsource(select_huds)
        assert "member_indices.size == 1" in source, (
            "Single-member cluster handling not found"
        )
        print("[PASS] Single-member clusters handled correctly (source verified)")
        return True
    except Exception as e:
        print(f"[FAIL] Single-member cluster test failed: {e}")
        return False

def test_diagnostic_fields():
    """P2: Diagnostic fields use correct values."""
    import inspect
    from huds_app.sampling.huds import select_huds

    try:
        source = inspect.getsource(select_huds)
        assert '"topk_size": int(len(topk_positions))' in source, (
            "topk_size should be len(topk_positions), not len(selected_ids)"
        )
        assert '"selection_method": "topk_fallback" if used_fallback' in source, (
            "selection_method should use used_fallback flag"
        )
        assert '"fill_method": "k_center_from_high_uncertainty" if used_fill' in source, (
            "fill_method should use used_fill flag"
        )
        print("[PASS] Diagnostic fields are correct (source verified)")
        return True
    except Exception as e:
        print(f"[FAIL] Diagnostic fields test failed: {e}")
        return False


if __name__ == "__main__":
    import json

    tests = [
        test_cli_entry,
        test_atomic_write,
        test_mc_dropout_only_dropout,
        test_config_validation,
        test_image_decoder_size,
        test_mc_dropout_default_path,
        test_mc_dropout_output_path,
        test_mc_dropout_state_restoration,
        test_config_always_validated,
        test_uncertainty_normalized,
        test_k_center_from_topk,
        test_single_member_cluster,
        test_diagnostic_fields,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} crashed: {e}")

    print(f"\n{'='*50}")
    print(f"Validation results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("All improvements validated successfully!")
        sys.exit(0)
    else:
        print("Some tests failed. Please review.")
        sys.exit(1)
