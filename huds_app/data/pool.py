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
    pool_df = pd.DataFrame(sample_array, columns=pd.Index(variable_names))
    pool_df.insert(0, "status", "unlabeled")
    pool_df.insert(0, "split", "validation_pool")
    pool_df.insert(0, "sample_id", np.arange(total_samples, dtype=int))

    train_df, valid_df = split_pool(pool_df, config, config.random_seed)
    return pd.concat([train_df, valid_df], ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def split_pool(df: pd.DataFrame, config: Any, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_samples = int(config.candidate_pool.total_samples)
    sample_ids = list(range(total_samples))
    random.Random(random_seed).shuffle(sample_ids)

    train_size = int(total_samples * float(config.candidate_pool.train_ratio))
    train_ids = sample_ids[:train_size]

    pool_df = df.copy()
    pool_df["split"] = np.where(pool_df["sample_id"].isin(train_ids), "train_pool", "validation_pool")

    train_df = pool_df.loc[pool_df["split"] == "train_pool"].copy().reset_index(drop=True)
    valid_df = pool_df.loc[pool_df["split"] == "validation_pool"].copy().reset_index(drop=True)
    return train_df, valid_df


def save_pool_files(pool_df: pd.DataFrame, train_df: pd.DataFrame, valid_df: pd.DataFrame, run_dir: str | Path) -> None:
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_columns = [column for column in pool_df.columns if column not in {"split", "status"}]

    pool_df.to_csv(output_dir / "candidate_pool.csv", index=False)
    train_df[variable_columns].to_csv(output_dir / "train_pool.csv", index=False)
    valid_df[variable_columns].to_csv(output_dir / "validation_pool.csv", index=False)


def _snap_column_to_levels(column: np.ndarray, min_value: float, max_value: float, sample_points: int) -> np.ndarray:
    levels = np.linspace(min_value, max_value, sample_points)
    indices = np.searchsorted(levels, column, side="left")
    indices = np.clip(indices, 0, sample_points - 1)

    previous_indices = np.clip(indices - 1, 0, sample_points - 1)
    previous_distances = np.abs(column - levels[previous_indices])
    next_distances = np.abs(column - levels[indices])
    nearest_indices = np.where(previous_distances <= next_distances, previous_indices, indices)
    return levels[nearest_indices]
