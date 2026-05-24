from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


_RUN_SUBDIRS = ("requests", "datasets", "checkpoints", "metrics", "artifacts")


@dataclass
class RunState:
    run_dir: str
    current_step: int = 0
    validation_request_created: bool = False
    validation_labeled: bool = False
    train_requests: dict[str, dict[str, str]] = field(default_factory=dict)
    latest_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    used_sample_ids: list[Any] = field(default_factory=list)
    pending_sample_ids: list[Any] = field(default_factory=list)

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

        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)
            file.write("\n")


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
        return pd.read_csv(csv_path)
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
