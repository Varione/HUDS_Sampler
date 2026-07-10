"""Installation-level smoke tests for HUDS app."""

import subprocess
import sys


def test_cli_entry_point():
    """Test that the CLI entry point is properly registered and can display help."""
    result = subprocess.run(
        [sys.executable, "-m", "huds_app.interface.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed with: {result.stderr}"
    assert "usage:" in result.stdout.lower()
    assert "init" in result.stdout
    assert "sample" in result.stdout
    assert "train" in result.stdout


def test_cli_help():
    """Test that the CLI help message displays correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "huds_app.interface.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "hudS active learning workflow" in result.stdout.lower() or "hudS" in result.stdout.lower()


if __name__ == "__main__":
    test_cli_entry_point()
    test_cli_help()
    print("All smoke tests passed!")
