from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch

from huds_app.core.config import inspect_config as _inspect_loaded_config
from huds_app.core.config import load_config

_logger = logging.getLogger(__name__)
from huds_app.data.schema import (
    SAMPLE_ID_COLUMN,
    candidate_pool,
    validate_schema,
    validate_values,
)
from huds_app.core.metrics import compute_metrics
from huds_app.model.architecture import build_model
from huds_app.data.pool import create_candidate_pool, save_pool_files
from huds_app.core.storage import RunState, _normalize_sample_id, ensure_run_dir, read_csv, resolve_device
from huds_app.model.train import apply_normalization, load_normalization


def init_run(config_path: str | Path, run_dir: str | Path, snap_to_levels: bool = False) -> dict[str, Any]:
    config = load_config(str(config_path))
    run_path = ensure_run_dir(str(run_dir))

    dst = run_path / "config.json"
    if Path(config_path).resolve() != dst.resolve():
        shutil.copy2(config_path, dst)

    pool_df = create_candidate_pool(config, snap_to_levels=snap_to_levels)
    save_pool_files(pool_df, run_path)

    state = RunState(run_dir=str(run_path))
    state.save()

    return {
        "run_dir": str(run_path),
        "total_candidates": int(len(pool_df)),
    }


def show_status(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    state = RunState.load(str(run_path))
    config = load_config(str(run_path / "config.json"))

    candidate_df = _read_optional_csv(run_path / "candidate_pool.csv")
    datasets_dir = run_path / "datasets"
    train_labeled_df = _read_optional_csv(datasets_dir / "train_labeled.csv")
    val_labeled_df = _read_optional_csv(datasets_dir / "val_labeled.csv")
    test_labeled_df = _read_optional_csv(datasets_dir / "test_labeled.csv")

    labeled_ids = set()
    if train_labeled_df is not None:
        labeled_ids.update(_sample_id_set(train_labeled_df))
    if val_labeled_df is not None:
        labeled_ids.update(_sample_id_set(val_labeled_df))
    if test_labeled_df is not None:
        labeled_ids.update(_sample_id_set(test_labeled_df))

    remaining_unlabeled = _remaining_unlabeled_count(candidate_df, labeled_ids, state)
    pending_count = len(state.pending_sample_ids)
    available_for_sampling = max(0, remaining_unlabeled - pending_count)
    latest_checkpoint = state.latest_checkpoint or _existing_relative_path(run_path, run_path / "checkpoints" / "model_latest.pt")

    status = {
        "total_candidates": _row_count(candidate_df),
        "labeled_train_count": _row_count(train_labeled_df),
        "labeled_val_count": _row_count(val_labeled_df),
        "labeled_test_count": _row_count(test_labeled_df),
        "current_step": int(state.current_step),
        "trained_step": int(state.trained_step),
        "max_steps": int(config.training.max_steps),
        "latest_checkpoint": latest_checkpoint,
        "remaining_unlabeled": remaining_unlabeled,
        "pending_count": pending_count,
        "available_for_sampling": available_for_sampling,
        "next_command": _next_command(
            state,
            train_labeled_df,
            val_labeled_df,
            latest_checkpoint,
            config,
            available_for_sampling,
        ),
    }

    _print_status(status, config.project_name)
    return status


def validate_files(run_dir: str | Path) -> List[str]:
    run_path = Path(run_dir)
    errors: List[str] = []
    expected_files = [
        run_path / "candidate_pool.csv",
        run_path / "state.json",
        run_path / "config.json",
    ]

    for path in expected_files:
        if not path.exists():
            errors.append(f"Missing expected file: {_display_path(path)}")
        elif not path.is_file():
            errors.append(f"Expected file path is not a file: {_display_path(path)}")

    config = _load_config_for_validation(run_path, errors)
    if config is None:
        return errors

    pool_specs = [
        (run_path / "candidate_pool.csv", candidate_pool, _request_columns(config)),
    ]
    valid_ids_by_file: dict[Path, set[Any]] = {}
    for path, schema, numeric_columns in pool_specs:
        df = _read_csv_for_validation(path, errors)
        if df is None:
            continue
        valid_ids_by_file[path] = _sample_id_set(df)
        errors.extend(_schema_errors(path, df, schema))
        errors.extend(_column_errors(path, df, numeric_columns))
        errors.extend(_duplicate_id_errors(path, df))

    labeled_specs = [
        (datasets_dir / "train_labeled.csv", valid_ids_by_file.get(run_path / "candidate_pool.csv", set())),
        (datasets_dir / "val_labeled.csv", valid_ids_by_file.get(run_path / "candidate_pool.csv", set())),
        (datasets_dir / "test_labeled.csv", valid_ids_by_file.get(run_path / "candidate_pool.csv", set())),
    ]
    datasets_dir = run_path / "datasets"
    for path, valid_ids in labeled_specs:
        if not path.exists():
            continue
        df = _read_csv_for_validation(path, errors)
        if df is None:
            continue
        required_columns = _labeled_columns(config)
        errors.extend(_column_errors(path, df, required_columns))
        errors.extend(_duplicate_id_errors(path, df))
        if valid_ids and SAMPLE_ID_COLUMN in df.columns:
            unknown_ids = sorted(set(df[SAMPLE_ID_COLUMN].dropna().tolist()) - valid_ids)
            if unknown_ids:
                errors.append(f"{_display_path(path)} contains unknown sample_id(s): {unknown_ids}")

    return errors


def inspect_config(config_path: str | Path) -> None:
    config = load_config(str(config_path))
    _inspect_loaded_config(config)


def predict(run_dir: str | Path, input_path: str | Path, output_path: str | Path) -> None:
    run_path = Path(run_dir)
    config = load_config(str(run_path / "config.json"))
    input_df = read_csv(input_path)

    variable_columns = _variable_columns(config)
    _require_columns(input_df, variable_columns, Path(input_path))

    model, device = _load_model_for_inference(run_path, config)
    normalization = load_normalization(run_path / "artifacts" / "normalization.json")
    predictions = _predict_array(model, device, input_df, variable_columns, normalization)
    predictions = _denormalize_predictions(predictions, config.model.output_names, normalization)

    output_df = input_df.copy()
    for index, output_name in enumerate(config.model.output_names):
        output_df[output_name] = predictions[:, index]
    write_csv(output_df, output_path)


def evaluate(run_dir: str | Path) -> dict[str, float]:
    run_path = Path(run_dir)
    config = load_config(str(run_path / "config.json"))
    test_df = read_csv(run_path / "datasets" / "test_labeled.csv")

    variable_columns = _variable_columns(config)
    output_columns = list(config.model.output_names)
    _require_columns(test_df, [*variable_columns, *output_columns], run_path / "datasets" / "test_labeled.csv")

    model, device = _load_model_for_inference(run_path, config)
    normalization = load_normalization(run_path / "artifacts" / "normalization.json")
    predictions = _predict_array(model, device, test_df, variable_columns, normalization)
    predictions = _denormalize_predictions(predictions, output_columns, normalization)
    y_true = test_df[output_columns].to_numpy(dtype=float)

    return compute_metrics(y_true, predictions, output_columns)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return read_csv(path) if path.exists() else None


def _row_count(df: pd.DataFrame | None) -> int:
    return 0 if df is None else int(len(df))


def _sample_id_set(df: pd.DataFrame | None) -> set[Any]:
    if df is None or SAMPLE_ID_COLUMN not in df.columns:
        return set()
    return set(df[SAMPLE_ID_COLUMN].dropna().tolist())


def _remaining_unlabeled_count(candidate_df: pd.DataFrame | None, labeled_ids: set[Any], state: RunState) -> int:
    if candidate_df is None or SAMPLE_ID_COLUMN not in candidate_df.columns:
        return 0
    blocked_ids = set(state.used_sample_ids) | set(state.pending_sample_ids) | labeled_ids
    return int((~candidate_df[SAMPLE_ID_COLUMN].isin(list(blocked_ids))).sum())


def _existing_relative_path(run_path: Path, path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.relative_to(run_path).as_posix()
    except ValueError:
        return str(path)


def _next_command(
    state: RunState,
    train_labeled_df: pd.DataFrame | None,
    val_labeled_df: pd.DataFrame | None,
    latest_checkpoint: str | None,
    config: Any,
    available_for_sampling: int,
) -> str:
    """Determine the next recommended command based on workflow state."""
    # Phase 1: find first step with exported/partial status (needs label import)
    pending_steps = []
    for step_str in sorted(state.train_requests.keys(), key=int):
        req = state.train_requests[step_str]
        status = req.get("status")
        if status is None:
            _logger.warning(
                "train_request step %s has no 'status' field, treating as unknown",
                step_str,
            )
            continue
        if status in ("exported", "partial"):
            pending_steps.append(step_str)

    if pending_steps:
        first_pending = pending_steps[0]
        return f"import-labels --step {first_pending}"

    # Phase 2: all requests labeled, check if training is needed
    max_trained_step = int(getattr(state, "trained_step", -1))
    labeled_steps = [
        int(k) for k, v in state.train_requests.items()
        if v.get("status") == "labeled"
    ]
    if not labeled_steps:
        return "sample --step 1"

    max_labeled_step = max(labeled_steps)
    if max_trained_step < max_labeled_step:
        return "train"

    # Phase 3: training complete, check sampling eligibility
    if int(state.current_step) >= int(config.training.max_steps):
        return ""
    if available_for_sampling <= 0:
        return ""

    next_step = int(state.current_step) + 1
    return f"sample --step {next_step}"


def _print_status(status: dict[str, Any], project_name: str) -> None:
    if project_name:
        print(f"Project: {project_name}")
    for key, value in status.items():
        print(f"{key}: {value}")


def _load_config_for_validation(run_path: Path, errors: List[str]) -> Any | None:
    config_path = run_path / "config.json"
    if not config_path.exists() or not config_path.is_file():
        return None
    try:
        return load_config(str(config_path))
    except Exception as error:  # noqa: BLE001
        errors.append(f"Invalid config.json: {error}")
        return None


def _read_csv_for_validation(path: Path, errors: List[str]) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return read_csv(path)
    except Exception as error:  # noqa: BLE001
        errors.append(f"Invalid CSV {_display_path(path)}: {error}")
        return None


def _schema_errors(path: Path, df: pd.DataFrame, schema: Any) -> list[str]:
    return [f"{_display_path(path)} missing required column: {column}" for column in validate_schema(df, schema)]


def _column_errors(path: Path, df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    errors: list[str] = []
    missing = [column for column in required_columns if column not in df.columns]
    for column in missing:
        errors.append(f"{_display_path(path)} missing required column: {column}")

    present_numeric_columns = [column for column in required_columns if column in df.columns]
    for error in validate_values(df, present_numeric_columns):
        errors.append(f"{_display_path(path)} {error}")
    return errors


def _duplicate_id_errors(path: Path, df: pd.DataFrame) -> list[str]:
    if SAMPLE_ID_COLUMN not in df.columns:
        return []
    duplicates = df[SAMPLE_ID_COLUMN][df[SAMPLE_ID_COLUMN].duplicated()].dropna().unique()
    if len(duplicates) == 0:
        return []
    return [f"{_display_path(path)} contains duplicate sample_id(s): {sorted(duplicates.tolist())}"]


def _variable_columns(config: Any) -> list[str]:
    return [variable.name for variable in config.variables]


def _request_columns(config: Any) -> list[str]:
    return [SAMPLE_ID_COLUMN, *_variable_columns(config)]


def _labeled_columns(config: Any) -> list[str]:
    return [*_request_columns(config), *config.model.output_names]


def _require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def _load_model_for_inference(run_dir: str | Path, config: Any) -> tuple[Any, torch.device]:
    run_path = Path(run_dir)
    checkpoint_path = run_path / "checkpoints" / "model_latest.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = resolve_device(config.training.device)
    model = build_model(config).to(device)

    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict"))
        if state_dict is None:
            raise ValueError(f"Checkpoint {checkpoint_path} has no model state dict")
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: strict checkpoint load failed ({e}), attempting non-strict load...")
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint_data.get("model_state_dict", checkpoint_data.get("state_dict"))
            if state_dict is None:
                raise ValueError(f"Checkpoint {checkpoint_path} has no model state dict")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {error}")

    return model, device


def _predict_array(model: Any, device: torch.device, df: pd.DataFrame, var_cols: list[str], normalization: dict[str, Any]) -> np.ndarray:
    x_columns = []
    for column in var_cols:
        values = df[column].to_numpy(dtype=np.float32)
        normalized_values = apply_normalization(values, normalization["variables"][column])
        x_columns.append(normalized_values)

    input_array = np.column_stack(x_columns).astype(np.float32)
    input_tensor = torch.from_numpy(input_array).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)

    return predictions.cpu().numpy()


def _denormalize_predictions(predictions: np.ndarray, output_names: list[str], normalization: dict[str, Any]) -> np.ndarray:
    denormalized = np.zeros_like(predictions)
    for i, name in enumerate(output_names):
        stats = normalization["outputs"][name]
        mean_val = float(stats["mean"])
        std_val = float(stats["std"])
        if std_val > 0:
            denormalized[:, i] = predictions[:, i] * std_val + mean_val
    return denormalized
