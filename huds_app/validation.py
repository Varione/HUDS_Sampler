"""Validation request export and label import helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import AppConfig, load_config
from .data_schema import (
    SAMPLE_ID_COLUMN,
    simulator_output,
    validate_sample_ids,
    validate_schema,
    validate_values,
)
from .storage import RunState, append_csv, ensure_run_dir, read_csv, write_csv

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


def _select_request_rows(pool: pd.DataFrame, size: int, pool_name: str) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("request size must be > 0")
    if size > len(pool):
        print(
            f"Warning: requested {size} {pool_name} sample(s), "
            f"but only {len(pool)} available; using all available samples."
        )
        size = len(pool)
    return pool.head(size).copy()


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    column = df[name]
    if isinstance(column, pd.DataFrame):
        raise ValueError(f"CSV contains duplicate column name: {name}")
    return column


def _duplicate_values(series: pd.Series) -> list[object]:
    duplicate_mask = series.duplicated(keep=False)
    return series.loc[duplicate_mask].dropna().unique().tolist()


def _validate_request_source(pool: pd.DataFrame, config: AppConfig, source_name: str) -> None:
    missing = [column for column in _request_columns(config) if column not in pool.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required column(s): {missing}")

    duplicate_ids = _duplicate_values(_column(pool, SAMPLE_ID_COLUMN))
    if duplicate_ids:
        raise ValueError(f"{source_name} contains duplicate sample_id(s): {duplicate_ids}")


def _register_request_ids(config: AppConfig, request_df: pd.DataFrame, replace: bool = False) -> None:
    request_ids = set(_column(request_df, SAMPLE_ID_COLUMN).tolist())
    if replace:
        _REQUEST_IDS_BY_CONFIG[id(config)] = request_ids
        return
    _REQUEST_IDS_BY_CONFIG.setdefault(id(config), set()).update(request_ids)


def _request_path_for_import(run_path: Path, kind: str, step: int | str | None) -> Path:
    if kind == "validation":
        return run_path / "requests" / "validation_request.csv"
    if kind == "train":
        if step is None:
            raise ValueError("step is required when importing train labels")
        return run_path / "requests" / f"train_step_{int(step):03d}_request.csv"
    raise ValueError("kind must be either 'validation' or 'train'")


def _prepare_output_for_validation(config: AppConfig, request_path: Path) -> None:
    request_df = read_csv(request_path)
    _register_request_ids(config, request_df, replace=True)


def export_validation_request(run_dir: str | Path, config: AppConfig, size: int | None = None) -> Path:
    """Export validation samples for external simulation.
    
    FIX 1: Don't add to global pending_sample_ids - validation labels are tracked separately.
    """
    run_path = ensure_run_dir(str(run_dir))
    pool = read_csv(run_path / "validation_pool.csv")
    _validate_request_source(pool, config, "validation_pool.csv")

    request_size = config.validation.default_size if size is None else size
    request_df = _select_request_rows(pool[_request_columns(config)], request_size, "validation")
    output_path = run_path / "requests" / "validation_request.csv"
    write_csv(request_df, output_path)

    state = _load_state(run_path)
    state.validation_request_created = True
    # FIX 1: Don't add validation IDs to pending_sample_ids - they're tracked via validation_labeled flag
    state.save()

    _register_request_ids(config, request_df, replace=True)
    return output_path


def export_initial_train_request(run_dir: str | Path, config: AppConfig) -> Path:
    """Export initial training samples for external simulation."""
    run_path = ensure_run_dir(str(run_dir))
    pool = read_csv(run_path / "train_pool.csv")
    _validate_request_source(pool, config, "train_pool.csv")

    request_df = _select_request_rows(
        pool[_request_columns(config)],
        config.training.initial_train_size,
        "initial training",
    )
    output_path = run_path / "requests" / "train_step_000_request.csv"
    write_csv(request_df, output_path)

    state = _load_state(run_path)
    state.train_requests["0"] = {
        "path": _relative_to_run(run_path, output_path),
        "status": "exported",
    }
    # Add to pending_sample_ids for training requests (FIX 1: proper tracking)
    request_ids = list(request_df[SAMPLE_ID_COLUMN].tolist())
    state.pending_sample_ids.extend(request_ids)
    state.save()

    _register_request_ids(config, request_df, replace=True)
    return output_path


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
    """Validate that all requested IDs are covered (existing + incoming).

    Returns the set of still-missing IDs (empty if fully complete).
    Raises ValueError if incomplete and allow_partial is False.

    Cumulative partial logic: existing labeled data counts toward completeness
    so multiple --allow-partial imports can collectively satisfy a step request.
    """
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


def _update_import_state(
    run_path: Path,
    kind: str,
    step_key: str | None,
    imported_ids: list[int],
    pending_missing: set[object] | None,
) -> None:
    """Update state after importing labels.

    Called AFTER _validate_import_completeness and AFTER files are written.
    pending_missing is the missing-IDs set returned by validation (empty = complete).
    """
    state = _load_state(run_path)
    if kind == "validation":
        state.validation_labeled = True
    elif kind == "train" and step_key is not None:
        is_complete = pending_missing is None or len(pending_missing) == 0

        if is_complete:
            state.train_requests.setdefault(step_key, {})["status"] = "labeled"
        else:
            state.train_requests[step_key] = {
                **state.train_requests.get(step_key, {}),
                "status": "partial",
            }

        # Clear pending IDs only for the portion that is now labeled.
        if is_complete:
            imported_set = set(imported_ids)
            state.pending_sample_ids = [
                sid for sid in state.pending_sample_ids if sid not in imported_set
            ]

    state.save()


def import_labels(
    run_dir: str | Path,
    kind: str,
    step: int | str | None,
    input_path: str | Path,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> int:
    """Import simulator output labels.

    P1 FIX A  – All validation runs BEFORE any file writes so a failed import
                never corrupts existing labeled data on disk.
    P1 FIX B  – Completeness check uses cumulative IDs (existing labeled CSV +
                incoming), so multiple --allow-partial imports can collectively
                satisfy a step request and flip status to "labeled".

    Args:
        run_dir: Run directory path
        kind: 'validation' or 'train'
        step: Training step number (required for train kind)
        input_path: Path to simulator output CSV
        overwrite: Whether to overwrite existing labeled data
        allow_partial: Whether to permit partial label imports
    """
    run_path = ensure_run_dir(str(run_dir))
    config = _load_config(run_path)
    request_path = _request_path_for_import(run_path, kind, step)

    # --- 1. Read inputs (read-only phase) ---
    request_df = read_csv(request_path)
    _register_request_ids(config, request_df, replace=True)

    simulator_df = read_csv(input_path)
    errors = validate_simulator_output(simulator_df, config)
    if errors:
        raise ValueError("Invalid simulator output: " + "; ".join(errors))

    labeled_df = simulator_df[_labeled_columns(config)].copy()
    incoming_ids = set(_column(labeled_df, SAMPLE_ID_COLUMN).tolist())

    # --- 2. Read existing labeled data for cumulative check (P1 FIX B) ---
    #     If overwrite=True, the file will be replaced so only incoming counts.
    output_path = run_path / "datasets" / f"{kind}_labeled.csv"
    if overwrite or not output_path.exists():
        existing_labeled_ids: set[object] = set()
    else:
        existing_labeled_ids = set(read_csv(output_path)[SAMPLE_ID_COLUMN].tolist())

    # --- 3. Validate completeness BEFORE any file writes (P1 FIX A) ---
    step_key = str(int(step)) if kind == "train" and step is not None else None
    requested_ids = set(_column(request_df, SAMPLE_ID_COLUMN).tolist())
    pending_missing = _validate_import_completeness(
        requested_ids, existing_labeled_ids, incoming_ids, allow_partial, step_key
    )

    # --- 4. Write files (only after all validation passed) ---
    imported_ids_list = list(labeled_df[SAMPLE_ID_COLUMN].tolist())
    _write_labeled_data(output_path, labeled_df, overwrite)

    # --- 5. Update state ---
    _update_import_state(run_path, kind, step_key, imported_ids_list, pending_missing)

    return len(labeled_df)
