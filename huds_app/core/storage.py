from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

# Try to import filelock; if not available, create a dummy lock class
try:
    from filelock import FileLock
except ImportError:
    # Dummy implementation for systems without filelock
    class FileLock:
        def __init__(self, lock_file: str | Path, timeout: float = 10):
            self.lock_path = Path(lock_file)
            self.timeout = timeout

        def acquire(self):
            while not self.lock_path.exists():
                try:
                    self.lock_path.touch()
                    return
                except FileExistsError:
                    pass
                import time
                time.sleep(0.1)

        def release(self):
            try:
                self.lock_path.unlink()
            except OSError:
                pass

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()


def resolve_device(config_device: str) -> torch.device:
    """Resolve torch device based on config and hardware availability.

    Priority:
    1. CUDA if configured and available
    2. MPS if configured and available (macOS)
    3. Config value as fallback (with warning if unavailable)
    4. CPU as final fallback
    """
    device_str = config_device.lower().strip()

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # CUDA requested but not available -> fallback to CPU
    if "cuda" in device_str and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but unavailable, falling back to CPU")
        return torch.device("cpu")

    # MPS requested but not available -> fallback to CPU
    if "mps" in device_str and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        print(f"Warning: MPS requested but unavailable, falling back to CPU")
        return torch.device("cpu")

    # Generic fallback: try config value directly
    try:
        return torch.device(device_str)
    except RuntimeError:
        print(f"Warning: device '{config_device}' not available, falling back to CPU")
        return torch.device("cpu")


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using temporary file and replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _normalize_sample_id(value: Any) -> int | str:
    """Normalize sample_id to consistent int or str type.

    Prevents type mismatches when mixing pandas int64, float64, and Python int
    across set operations and .isin() lookups.
    """
    import numpy as np
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError(f"sample_id is NaN: {value}")
        # Use Decimal to avoid precision loss for large integers (>2^53)
        decimal_val = Decimal(str(value))
        int_val = int(decimal_val.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        if decimal_val == Decimal(int_val):
            return int_val
        return str(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value)


def run_directory_lock(run_dir: str | Path, timeout: float = 300) -> FileLock:
    """Return a file lock for the run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    if not run_path.is_dir():
        raise ValueError(f"Run path is not a directory: {run_path}")

    lock_file = run_path / ".huds.lock"
    return FileLock(str(lock_file), timeout=timeout)


_RUN_SUBDIRS = ("requests", "datasets", "checkpoints", "metrics", "artifacts")


@dataclass
class RunState:
    run_dir: str
    current_step: int = 0
    train_requests: dict[str, dict[str, str]] = field(default_factory=dict)
    latest_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    trained_step: int = -1
    used_sample_ids: list[Any] = field(default_factory=list)
    pending_sample_ids: list[Any] = field(default_factory=list)
    split_assignments: dict[str, dict[str, list]] = field(default_factory=dict)

    @property
    def state_path(self) -> Path:
        return Path(self.run_dir) / "state.json"

    @classmethod
    def load(cls, run_dir: str) -> "RunState":
        state_path = Path(run_dir) / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Run state file not found: {state_path}")
        if not state_path.is_file():
            raise ValueError(f"Run state path is not a file: {state_path}")

        try:
            # FIX 4: Use utf-8-sig encoding to handle UTF-8 BOM files from Windows/PowerShell
            with state_path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in run state file {state_path}: {error}") from error

        if not isinstance(data, dict):
            raise ValueError(f"Run state file must contain a JSON object: {state_path}")

        data.setdefault("run_dir", str(run_dir))
        return cls(**data)

    def save(self) -> None:
        run_path = Path(self.run_dir)
        if run_path.exists() and not run_path.is_dir():
            raise ValueError(f"Run directory path is not a directory: {run_path}")
        run_path.mkdir(parents=True, exist_ok=True)

        # Use atomic write to prevent corruption on interruption
        atomic_write_json(self.state_path, asdict(self))


def ensure_run_dir(run_dir: str) -> Path:
    run_path = Path(run_dir)
    if run_path.exists() and not run_path.is_dir():
        raise ValueError(f"Run directory path is not a directory: {run_path}")

    run_path.mkdir(parents=True, exist_ok=True)
    for subdir in _RUN_SUBDIRS:
        (run_path / subdir).mkdir(exist_ok=True)
    return run_path


def list_request_files(run_dir: str) -> list[Path]:
    requests_dir = Path(run_dir) / "requests"
    if not requests_dir.exists():
        raise FileNotFoundError(f"Requests directory not found: {requests_dir}")
    if not requests_dir.is_dir():
        raise ValueError(f"Requests path is not a directory: {requests_dir}")
    return sorted(path for path in requests_dir.glob("*.csv") if path.is_file())


def load_run_config(run_dir: str) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Run config path is not a file: {config_path}")

    try:
        # FIX 4: Use utf-8-sig encoding to handle UTF-8 BOM files from Windows/PowerShell
        with config_path.open("r", encoding="utf-8-sig") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in run config file {config_path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"Run config file must contain a JSON object: {config_path}")
    return config


def read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")

    try:
        # Read sample_id as string to preserve precision for large integers (>2^53).
        # pandas default float64 truncates values beyond 2^53.
        df = pd.read_csv(csv_path, dtype={"sample_id": str})
        return df
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"CSV file is empty: {csv_path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"CSV file could not be parsed: {csv_path}: {error}") from error


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("write_csv expects a pandas DataFrame")

    csv_path = Path(path)
    if csv_path.exists() and not csv_path.is_file():
        raise ValueError(f"CSV output path is not a file: {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("append_csv expects a pandas DataFrame")

    csv_path = Path(path)
    if csv_path.exists() and not csv_path.is_file():
        raise ValueError(f"CSV output path is not a file: {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)




