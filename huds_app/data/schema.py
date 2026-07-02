"""CSV schema validation helpers for the HUDS active learning app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Description of a CSV column used by a schema."""

    name: str
    required: bool = True


@dataclass(frozen=True)
class SchemaDefinition:
    """Named collection of expected CSV columns."""

    name: str
    columns: List[ColumnSpec]


SAMPLE_ID_COLUMN = "sample_id"
SPLIT_COLUMN = "split"
STATUS_COLUMN = "status"

KNOWN_METADATA_COLUMNS = [SAMPLE_ID_COLUMN, SPLIT_COLUMN, STATUS_COLUMN]
VALID_SPLITS = ["train_pool", "validation_pool"]
VALID_STATUSES = ["unlabeled", "selected", "labeled", "used"]


candidate_pool = SchemaDefinition(
    name="candidate_pool",
    columns=[
        ColumnSpec(SAMPLE_ID_COLUMN),
        ColumnSpec(SPLIT_COLUMN),
        ColumnSpec(STATUS_COLUMN),
    ],
)
train_pool = SchemaDefinition(
    name="train_pool",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)
validation_pool = SchemaDefinition(
    name="validation_pool",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)
validation_request = SchemaDefinition(
    name="validation_request",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)
train_request = SchemaDefinition(
    name="train_request",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)
simulator_output = SchemaDefinition(
    name="simulator_output",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)
labeled_dataset = SchemaDefinition(
    name="labeled_dataset",
    columns=[ColumnSpec(SAMPLE_ID_COLUMN)],
)


SCHEMAS: Dict[str, SchemaDefinition] = {
    schema.name: schema
    for schema in [
        candidate_pool,
        train_pool,
        validation_pool,
        validation_request,
        train_request,
        simulator_output,
        labeled_dataset,
    ]
}


def get_schema(name: str) -> Optional[SchemaDefinition]:
    """Return a predefined schema by name, or None when it is unknown."""

    return SCHEMAS.get(name)


def validate_schema(df: pd.DataFrame, schema: SchemaDefinition) -> List[str]:
    """Return missing required columns for a dataframe and schema."""

    return [
        column.name
        for column in schema.columns
        if column.required and column.name not in df.columns
    ]


def validate_values(df: pd.DataFrame, required_columns: Sequence[str]) -> List[str]:
    """Validate that required columns contain present finite numeric values."""

    errors: List[str] = []
    for column in required_columns:
        if column not in df.columns:
            errors.append(f"Missing required column: {column}")
            continue

        values = df[column]
        nan_count = int(values.isna().sum())
        if nan_count:
            errors.append(f"Column {column} contains {nan_count} missing value(s)")

        numeric_values = pd.to_numeric(values, errors="coerce")
        non_numeric_count = int(numeric_values.isna().sum() - values.isna().sum())
        if non_numeric_count:
            errors.append(f"Column {column} contains {non_numeric_count} non numeric value(s)")

        finite_mask = np.isfinite(numeric_values.to_numpy(dtype=float, na_value=np.nan))
        infinite_count = int((~finite_mask & numeric_values.notna().to_numpy()).sum())
        if infinite_count:
            errors.append(f"Column {column} contains {infinite_count} infinite value(s)")

    return errors


def validate_sample_ids(df: pd.DataFrame, valid_ids: Iterable[object]) -> List[str]:
    """Validate sample_id uniqueness and membership in the supplied id set."""

    errors: List[str] = []
    if SAMPLE_ID_COLUMN not in df.columns:
        return [f"Missing required column: {SAMPLE_ID_COLUMN}"]

    sample_ids = df[SAMPLE_ID_COLUMN]
    missing_count = int(sample_ids.isna().sum())
    if missing_count:
        errors.append(f"Column {SAMPLE_ID_COLUMN} contains {missing_count} missing value(s)")

    duplicate_values = sample_ids[sample_ids.duplicated(keep=False)].dropna().unique().tolist()
    if duplicate_values:
        errors.append(f"Column {SAMPLE_ID_COLUMN} contains duplicate id(s): {duplicate_values}")

    valid_id_set = set(valid_ids)
    invalid_values = sample_ids[~sample_ids.isin(valid_id_set)].dropna().unique().tolist()
    if invalid_values:
        errors.append(f"Column {SAMPLE_ID_COLUMN} contains unknown id(s): {invalid_values}")

    return errors


def infer_variable_columns(df: pd.DataFrame, exclude_cols: Iterable[str]) -> List[str]:
    """Infer variable columns by excluding known metadata or output columns."""

    excluded = set(exclude_cols)
    return [column for column in df.columns if column not in excluded]
