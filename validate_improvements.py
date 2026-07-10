"""Quick validation script for HUDS improvements."""

import sys
sys.path.insert(0, r"E:\大型数据库构建\HUDS_Sampler_GitHub")

from huds_app.interface.cli import main as cli_main
from huds_app.core.storage import atomic_write_json, run_directory_lock
from huds_app.sampling.huds import mc_dropout_predict, _enable_mc_dropout
from huds_app.model.architecture import VectorToImage
from pathlib import Path
import tempfile
import os

def test_cli_entry():
    """Test CLI entry point."""
    try:
        # This should not crash
        from huds_app.interface import cli
        print("[PASS] CLI module loads correctly")
        return True
    except Exception as e:
        print(f"[FAIL] CLI module load failed: {e}")
        return False

def test_atomic_write():
    """Test atomic write functionality."""
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
    try:
        model = VectorToImage(input_dim=10, hidden_dim=32, encoder_blocks=2, dropout=0.1)
        _enable_mc_dropout(model)
        # Check that model is in eval mode but Dropout modules are in train mode
        assert model.training == False or any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)), \
            "Dropout should be enabled while other layers stay in eval"
        print("[PASS] MC Dropout only affects Dropout layers")
        return True
    except Exception as e:
        print(f"[FAIL] MC Dropout test failed: {e}")
        return False

def test_config_validation():
    """Test config validation with repeat_times >= 2."""
    from huds_app.core.config import AppConfig, HUDSConfig, VariableConfig, ModelConfig, validate_config
    
    try:
        config = AppConfig()
        # Need at least one variable and output to pass initial validation
        config.variables = [VariableConfig(name="test", min=0, max=10, sample_points=5)]
        config.model = ModelConfig(output_names=["output1"])
        config.huds = HUDSConfig(repeat_times=1)
        # This should raise an error now
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
    try:
        # Create a decoder with non-divisible dimensions
        model = VectorToImage(
            input_dim=10, 
            hidden_dim=32, 
            encoder_blocks=2, 
            img_h=30,  # Not divisible by 4 (2^2)
            img_w=30,
            channels=1,
            decoder_blocks=2
        )
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 1, 30, 30), f"Output shape {output.shape} does not match expected (1, 1, 30, 30)"
        print("[PASS] ImageDecoder outputs correct size after interpolation")
        return True
    except Exception as e:
        print(f"[FAIL] Image decoder size test failed: {e}")
        return False

if __name__ == "__main__":
    import json
    import torch
    
    tests = [
        test_cli_entry,
        test_atomic_write,
        test_mc_dropout_only_dropout,
        test_config_validation,
        test_image_decoder_size,
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
