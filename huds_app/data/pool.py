from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_candidate_pool(config: Any, snap_to_levels: bool = False) -> pd.DataFrame:
    total_samples = int(config.candidate_pool.total_samples)
    rng = random.Random(config.random_seed)

    values_by_variable = []
    variable_names: list[str] = []

    for variable in config.variables:
        variable_names.append(variable.name)
        min_value = float(variable.min)
        max_value = float(variable.max)
        stratum_width = (max_value - min_value) / total_samples

        values = [
            rng.uniform(
                min_value + stratum_index * stratum_width,
                min_value + (stratum_index + 1) * stratum_width,
            )
            for stratum_index in range(total_samples)
        ]
        rng.shuffle(values)

        column = np.array(values, dtype=float)
        if snap_to_levels:
            column = _snap_column_to_levels(column, min_value, max_value, int(variable.sample_points))

        values_by_variable.append(column)

    sample_array = np.column_stack(values_by_variable) if values_by_variable else np.empty((total_samples, 0))
    # Simulation inputs are persisted and dispatched with a fixed precision so
    # candidate-pool values, AEDT requests, and diagnostics stay reproducible.
    sample_array = np.round(sample_array, decimals=5)
    pool_df = pd.DataFrame(sample_array, columns=pd.Index(variable_names))
    if variable_names and pool_df.duplicated(subset=variable_names).any():
        duplicate_count = int(pool_df.duplicated(subset=variable_names).sum())
        raise ValueError(
            f"Rounding to 5 decimal places produced {duplicate_count} duplicate parameter sets. "
            "Increase the parameter range or reduce candidate_pool.total_samples."
        )
    pool_df.insert(0, "status", "unlabeled")
    pool_df.insert(0, "sample_id", np.arange(total_samples, dtype=int))
    return pool_df


def save_pool_files(pool_df: pd.DataFrame, run_dir: str | Path) -> None:
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_columns = [column for column in pool_df.columns if column not in {"status"}]

    pool_df.to_csv(output_dir / "candidate_pool.csv", index=False, float_format="%.5f")


def _snap_column_to_levels(column: np.ndarray, min_value: float, max_value: float, sample_points: int) -> np.ndarray:
    levels = np.linspace(min_value, max_value, sample_points)
    indices = np.searchsorted(levels, column, side="left")
    indices = np.clip(indices, 0, sample_points - 1)

    previous_indices = np.clip(indices - 1, 0, sample_points - 1)
    previous_distances = np.abs(column - levels[previous_indices])
    next_distances = np.abs(column - levels[indices])
    nearest_indices = np.where(previous_distances <= next_distances, previous_indices, indices)
    return levels[nearest_indices]
