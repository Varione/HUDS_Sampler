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
    step: int | str | None, 
    imported_ids: list[int],
    request_df: pd.DataFrame,
    allow_partial: bool = False
) -> None:
    """Update state after importing labels.
    
    FIX 1: Properly manage pending_sample_ids when imports complete.
    FIX 3: Validate completeness of label import against request file.
    """
    # FIX 3: Check if all requested IDs were imported
    requested_ids = set(request_df[SAMPLE_ID_COLUMN].tolist())
    imported_id_set = set(imported_ids)
    
    missing_ids = requested_ids - imported_id_set
    if missing_ids and not allow_partial:
        raise ValueError(
            f"Import incomplete: {len(missing_ids)} sample IDs from request are missing in simulator output. "
            f"Use --allow-partial to permit partial imports, or ensure all requests were simulated."
        )
    
    state = _load_state(run_path)
    if kind == "validation":
        state.validation_labeled = True
    elif kind == "train":
        step_key = str(int(step)) if step is not None else "0"
        
        # FIX 3: Set status based on completeness
        if allow_partial and missing_ids:
            state.train_requests.setdefault(step_key, {})["status"] = "partial"
        else:
            state.train_requests.setdefault(step_key, {})["status"] = "labeled"
            
        # FIX 1: Clear pending IDs for this step after successful import
        if not allow_partial or len(missing_ids) == 0:
            ids_to_remove = set(requested_ids) & set(state.pending_sample_ids)
            state.pending_sample_ids = [id for id in state.pending_sample_ids if id not in ids_to_remove]
    
    state.save()


def import_labels(
    run_dir: str | Path,
    kind: str,
    step: int | str | None,
    input_path: str | Path,
    overwrite: bool = False,
    allow_partial: bool = False,  # FIX 3: New parameter for partial imports
) -> int:
    """Import simulator output labels.
    
    Args:
        run_dir: Run directory path
        kind: 'validation' or 'train'
        step: Training step number (required for train kind)
        input_path: Path to simulator output CSV
        overwrite: Whether to overwrite existing labeled data
        allow_partial: Whether to permit partial label imports (FIX 3)
    """
    run_path = ensure_run_dir(str(run_dir))
    config = _load_config(run_path)
    request_path = _request_path_for_import(run_path, kind, step)
    
    # Read the request file for completeness checking (FIX 3)
    request_df = read_csv(request_path)
    _register_request_ids(config, request_df, replace=True)

    simulator_df = read_csv(input_path)
    errors = validate_simulator_output(simulator_df, config)
    if errors:
        raise ValueError("Invalid simulator output: " + "; ".join(errors))

    labeled_df = simulator_df[_labeled_columns(config)].copy()
    imported_ids = list(labeled_df[SAMPLE_ID_COLUMN].tolist())
    
    output_path = run_path / "datasets" / f"{kind}_labeled.csv"
    _write_labeled_data(output_path, labeled_df, overwrite)
    
    # FIX 1 & 3: Update state with completeness checking and pending ID management
    _update_import_state(run_path, kind, step, imported_ids, request_df, allow_partial)
    
    return len(labeled_df)
