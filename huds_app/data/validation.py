"""Label import helpers for incremental three-set splitting."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from huds_app.core.config import AppConfig, load_config
from huds_app.data.schema import (
    SAMPLE_ID_COLUMN,
    SPLIT_ASSIGNMENT_COLUMN,
    VALID_SPLIT_ASSIGNMENTS,
    simulator_output,
    validate_sample_ids,
    validate_schema,
    validate_values,
)
from huds_app.core.storage import RunState, _normalize_sample_id, append_csv, ensure_run_dir, read_csv, write_csv

_REQUEST_IDS_BY_CONFIG: dict[int, set[object]] = {}


def _variable_columns(config: AppConfig) -> list[str]:
    return [variable.name for variable in config.variables]


def _request_columns(config: AppConfig) -> list[str]:
    return [SAMPLE_ID_COLUMN, *_variable_columns(config)]


def _labeled_columns(config: AppConfig) -> list[str]:
    return [*_request_columns(config), *config.model.output_names]


def _load_config(run_path: Path) -> AppConfig:
    return load_config(str(run_path / "config.json"))


def _load_state(run_path: Path) -> RunState:
    return RunState.load(str(run_path))


def _relative_to_run(run_path: Path, path: Path) -> str:
    return path.relative_to(run_path).as_posix()


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    column = df[name]
    if isinstance(column, pd.DataFrame):
        raise ValueError(f"CSV contains duplicate column name: {name}")
    return column


def _duplicate_values(series: pd.Series) -> list[object]:
    duplicate_mask = series.duplicated(keep=False)
    return series.loc[duplicate_mask].dropna().unique().tolist()


def _register_request_ids(config: AppConfig, request_df: pd.DataFrame, replace: bool = False) -> None:
    request_ids = {_normalize_sample_id(sid) for sid in _column(request_df, SAMPLE_ID_COLUMN).tolist()}
    if replace:
        _REQUEST_IDS_BY_CONFIG[id(config)] = request_ids
        return
    _REQUEST_IDS_BY_CONFIG.setdefault(id(config), set()).update(request_ids)


def _request_path_for_step(run_path: Path, step: int | str) -> Path:
    return run_path / "requests" / f"train_step_{int(step):03d}_request.csv"


def validate_simulator_output(df: pd.DataFrame, config: AppConfig) -> List[str]:
    errors: List[str] = []
    required_columns = _labeled_columns(config)

    for column in validate_schema(df, simulator_output):
        errors.append(f"Missing required column: {column}")

    missing_columns = [column for column in required_columns if column not in df.columns]
    for column in missing_columns:
        if column != SAMPLE_ID_COLUMN:
            errors.append(f"Missing required column: {column}")

    if SAMPLE_ID_COLUMN in df.columns:
        duplicate_values = _duplicate_values(_column(df, SAMPLE_ID_COLUMN))
        if duplicate_values:
            errors.append(f"Column {SAMPLE_ID_COLUMN} contains duplicate id(s): {duplicate_values}")

    present_numeric_columns = [column for column in required_columns if column in df.columns]
    errors.extend(validate_values(df, present_numeric_columns))

    request_ids = _REQUEST_IDS_BY_CONFIG.get(id(config))
    if request_ids is not None:
        errors.extend(validate_sample_ids(df, request_ids))

    return errors


def _reject_duplicate_labels(existing: pd.DataFrame, incoming: pd.DataFrame) -> None:
    incoming_ids = _column(incoming, SAMPLE_ID_COLUMN)
    existing_ids = _column(existing, SAMPLE_ID_COLUMN)
    duplicate_ids = incoming_ids.loc[incoming_ids.isin(existing_ids)].dropna().unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate sample_id(s) already exist in labeled dataset: {duplicate_ids}")


def _validate_import_completeness(
    requested_ids: set[object],
    existing_labeled_ids: set[object],
    incoming_ids: set[object],
    allow_partial: bool,
    step_key: str | None,
) -> set[object]:
    combined_ids = existing_labeled_ids | incoming_ids
    missing_ids = requested_ids - combined_ids

    if missing_ids and not allow_partial:
        label = f" (step {step_key})" if step_key else ""
        raise ValueError(
            f"Import incomplete{label}: {len(missing_ids)} sample ID(s) from request "
            f"are missing in simulator output. Use --allow-partial to permit partial "
            f"imports, or ensure all requests were simulated."
        )
    return missing_ids


def _write_labeled_data(output_path: Path, labeled_df: pd.DataFrame, overwrite: bool) -> None:
    if overwrite or not output_path.exists():
        write_csv(labeled_df, output_path)
        return

    existing = read_csv(output_path)
    _reject_duplicate_labels(existing, labeled_df)
    append_csv(labeled_df, output_path)


def import_labels(
    run_dir: str | Path,
    step: int | str,
    input_path: str | Path,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> int:
    """Import simulator output labels and route to train/val/test sets.

    Reads the split assignment from the request CSV (split_assignment column),
    then routes labeled samples to the appropriate dataset file.

    Args:
        run_dir: Run directory path
        step: Training step number
        input_path: Path to simulator output CSV
        overwrite: Whether to overwrite existing labeled data
        allow_partial: Whether to permit partial label imports
    """
    run_path = ensure_run_dir(str(run_dir))
    config = _load_config(run_path)

    # --- 1. Read request and simulator output ---
    request_path = _request_path_for_step(run_path, step)
    request_df = read_csv(request_path)
    _register_request_ids(config, request_df, replace=True)

    simulator_df = read_csv(input_path)
    errors = validate_simulator_output(simulator_df, config)
    if errors:
        raise ValueError("Invalid simulator output: " + "; ".join(errors))

    labeled_df = simulator_df[_labeled_columns(config)].copy()
    incoming_ids = {_normalize_sample_id(sid) for sid in _column(labeled_df, SAMPLE_ID_COLUMN).tolist()}
    step_key = str(int(step))
    requested_ids = {_normalize_sample_id(sid) for sid in _column(request_df, SAMPLE_ID_COLUMN).tolist()}

    # --- 2. Read existing labeled data for cumulative check ---
    datasets_dir = run_path / "datasets"
    existing_labeled_ids: set[object] = set()
    for split_name in ("train", "val", "test"):
        output_path = datasets_dir / f"{split_name}_labeled.csv"
        if overwrite or not output_path.exists():
            continue
        existing_labeled_ids.update(
            {_normalize_sample_id(sid) for sid in read_csv(output_path)[SAMPLE_ID_COLUMN].tolist()}
        )

    # --- 3. Validate completeness BEFORE any file writes ---
    pending_missing = _validate_import_completeness(
        requested_ids, existing_labeled_ids, incoming_ids, allow_partial, step_key
    )

    # --- 4. Route labeled data by split assignment ---
    if SPLIT_ASSIGNMENT_COLUMN in request_df.columns:
        split_map = dict(zip(
            _column(request_df, SAMPLE_ID_COLUMN),
            _column(request_df, SPLIT_ASSIGNMENT_COLUMN),
        ))
    else:
        # Fallback: all samples go to train if no split column (backward compat)
        split_map = {sid: "train" for sid in _column(request_df, SAMPLE_ID_COLUMN)}

    labeled_df["_split"] = labeled_df[SAMPLE_ID_COLUMN].map(split_map)
    for split_name in ("train", "val", "test"):
        subset = labeled_df[labeled_df["_split"] == split_name].drop(columns=["_split"])
        if subset.empty:
            continue
        output_path = datasets_dir / f"{split_name}_labeled.csv"
        _write_labeled_data(output_path, subset, overwrite)

    # --- 5. Update state ---
    state = _load_state(run_path)
    is_complete = len(pending_missing) == 0
    state.train_requests[step_key] = {
        "path": _relative_to_run(run_path, request_path),
        "status": "labeled" if is_complete else "partial",
    }

    if is_complete:
        remove_set = requested_ids
    else:
        remove_set = requested_ids - pending_missing

    state.pending_sample_ids = [
        sid for sid in state.pending_sample_ids if sid not in remove_set
    ]
    state.save()

    return len(labeled_df)
