"""Tests for data schema validation."""

import math

import numpy as np
import pandas as pd
import pytest

from huds_app.data.schema import (
    ColumnSpec,
    SAMPLE_ID_COLUMN,
    SchemaDefinition,
    candidate_pool,
    infer_variable_columns,
    validate_sample_ids,
    validate_schema,
    validate_values,
)


class TestValidateSchema:
    def test_missing_required_column(self):
        schema = SchemaDefinition(
            name="test",
            columns=[
                ColumnSpec("a"),
                ColumnSpec("b"),
            ],
        )
        df = pd.DataFrame({"a": [1, 2]})
        missing = validate_schema(df, schema)
        assert missing == ["b"]

    def test_no_missing_columns(self):
        schema = SchemaDefinition(
            name="test",
            columns=[ColumnSpec("x"), ColumnSpec("y")],
        )
        df = pd.DataFrame({"x": [1], "y": [2]})
        missing = validate_schema(df, schema)
        assert missing == []

    def test_optional_column_missing_is_ok(self):
        schema = SchemaDefinition(
            name="test",
            columns=[ColumnSpec("a"), ColumnSpec("b", required=False)],
        )
        df = pd.DataFrame({"a": [1]})
        missing = validate_schema(df, schema)
        assert missing == []

    def test_candidate_pool_missing_split_and_status(self):
        df = pd.DataFrame({SAMPLE_ID_COLUMN: [0, 1]})
        missing = validate_schema(df, candidate_pool)
        assert "split" in missing
        assert "status" in missing


class TestValidateValues:
    def test_nan_detected(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        errors = validate_values(df, ["x"])
        assert any("missing value" in e for e in errors)

    def test_inf_detected(self):
        df = pd.DataFrame({"x": [1.0, float("inf"), 3.0]})
        errors = validate_values(df, ["x"])
        assert any("infinite value" in e for e in errors)

    def test_negative_inf_detected(self):
        df = pd.DataFrame({"x": [1.0, float("-inf")]})
        errors = validate_values(df, ["x"])
        assert any("infinite value" in e for e in errors)

    def test_valid_values_no_errors(self):
        df = pd.DataFrame({"x": [1.0, 2.5, -3.0], "y": [0, 0, 1]})
        errors = validate_values(df, ["x", "y"])
        assert errors == []

    def test_missing_column_reported(self):
        df = pd.DataFrame({"a": [1, 2]})
        errors = validate_values(df, ["b"])
        assert any("Missing required column" in e for e in errors)


class TestValidateSampleIds:
    def test_duplicate_ids_detected(self):
        df = pd.DataFrame({SAMPLE_ID_COLUMN: [0, 1, 1, 2]})
        errors = validate_sample_ids(df, valid_ids=[0, 1, 2])
        assert any("duplicate" in e for e in errors)

    def test_unique_ids_no_error(self):
        df = pd.DataFrame({SAMPLE_ID_COLUMN: [0, 1, 2]})
        errors = validate_sample_ids(df, valid_ids=[0, 1, 2])
        assert not any("duplicate" in e for e in errors)

    def test_unknown_id_detected(self):
        df = pd.DataFrame({SAMPLE_ID_COLUMN: [0, 99]})
        errors = validate_sample_ids(df, valid_ids=[0, 1, 2])
        assert any("unknown id" in e for e in errors)

    def test_missing_column_returns_error(self):
        df = pd.DataFrame({"other": [1]})
        errors = validate_sample_ids(df, valid_ids=[0])
        assert len(errors) == 1


class TestInferVariableColumns:
    def test_basic_inference(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = infer_variable_columns(df, exclude_cols=["a", "b"])
        assert result == ["c"]

    def test_empty_exclude_returns_all(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = infer_variable_columns(df, exclude_cols=[])
        assert set(result) == {"x", "y"}

    def test_exclude_more_than_exist(self):
        df = pd.DataFrame({"a": [1]})
        result = infer_variable_columns(df, exclude_cols=["a", "b"])
        assert result == []
