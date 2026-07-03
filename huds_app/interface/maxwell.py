from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from huds_app.core.config import AppConfig, load_config
from huds_app.data.schema import SAMPLE_ID_COLUMN
from huds_app.core.storage import read_csv, write_csv


def export_maxwell_table(
    input_path: str | Path,
    output_path: str | Path,
    config: AppConfig | str | Path | None = None,
    units: dict[str, str] | None = None,
    index_column_name: str = "*",
) -> Path:
    request_df = read_csv(input_path)
    app_config = _resolve_config(config)

    variable_columns = _resolve_variable_columns(request_df, app_config)
    resolved_units = _resolve_units(variable_columns, app_config, units or {})

    output_df = pd.DataFrame()
    output_df[index_column_name] = range(1, len(request_df) + 1)

    for column in variable_columns:
        output_df[column] = [
            _format_maxwell_value(value, resolved_units[column])
            for value in request_df[column].tolist()
        ]

    write_csv(output_df, output_path)
    return Path(output_path)


def parse_unit_overrides(entries: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid unit override '{entry}'. Expected NAME=UNIT.")
        name, unit = entry.split("=", 1)
        name = name.strip()
        unit = unit.strip()
        if not name or not unit:
            raise ValueError(f"Invalid unit override '{entry}'. Expected NAME=UNIT.")
        overrides[name] = unit
    return overrides


def _resolve_config(config: AppConfig | str | Path | None) -> AppConfig | None:
    if config is None:
        return None
    if isinstance(config, AppConfig):
        return config
    if isinstance(config, (str, Path)):
        return load_config(str(config))
    return config


def _resolve_variable_columns(request_df: pd.DataFrame, config: AppConfig | None) -> list[str]:
    if config is not None:
        columns = [variable.name for variable in config.variables]
        missing = [column for column in columns if column not in request_df.columns]
        if missing:
            raise ValueError(f"Request CSV is missing configured variable column(s): {missing}")
        return columns

    return [column for column in request_df.columns if column != SAMPLE_ID_COLUMN]


def _resolve_units(
    variable_columns: list[str],
    config: AppConfig | None,
    unit_overrides: dict[str, str],
) -> dict[str, str]:
    units: dict[str, str] = {}

    config_units = {
        variable.name: variable.resolved_unit()
        for variable in (config.variables if config is not None else [])
        if variable.resolved_unit()
    }

    for column in variable_columns:
        unit = unit_overrides.get(column) or config_units.get(column)
        if not unit:
            raise ValueError(
                f"Missing unit for variable '{column}'. Set variable.unit in config or pass --unit {column}=UNIT."
            )
        units[column] = unit

    return units


def _format_maxwell_value(value: Any, unit: str) -> str:
    # Reject NaN / Inf before Decimal conversion (Decimal accepts them silently).
    import math
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                f"Cannot format non-numeric value '{value}' for Maxwell export"
            )

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Cannot format non-numeric value '{value}' for Maxwell export") from error

    # Double-check: Decimal may have produced NaN/Inf from string input.
    if not decimal_value.is_finite():
        raise ValueError(
            f"Cannot format non-numeric value '{value}' for Maxwell export"
        )

    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return f"{normalized}{unit}"
