"""Tests for core.storage module."""
from __future__ import annotations

import numpy as np
import pytest

from huds_app.core.storage import _normalize_sample_id


class TestNormalizeSampleIdLargeValues:
    """Verify _normalize_sample_id preserves precision for large integer IDs."""

    def test_large_int_preserved(self):
        sid = 9007199254740993  # 2^53 + 1
        result = _normalize_sample_id(sid)
        assert result == 9007199254740993
        assert isinstance(result, int)

    def test_large_np_int64_preserved(self):
        sid = np.int64(9007199254740993)
        result = _normalize_sample_id(sid)
        assert result == 9007199254740993
        assert isinstance(result, int)

    def test_large_float_recovered(self):
        sid = 9007199254740992.0  # 2^53, exactly representable as float
        result = _normalize_sample_id(sid)
        assert result == 9007199254740992
        assert isinstance(result, int)

    def test_large_np_float64_recovered(self):
        sid = np.float64(9007199254740992.0)
        result = _normalize_sample_id(sid)
        assert result == 9007199254740992
        assert isinstance(result, int)

    def test_non_integer_float_returns_str(self):
        sid = 3.141592653589793
        result = _normalize_sample_id(sid)
        assert not isinstance(result, int)

    def test_string_large_int(self):
        sid = "9007199254740993"
        result = _normalize_sample_id(sid)
        assert result == 9007199254740993
        assert isinstance(result, int)

    def test_string_non_numeric(self):
        sid = "SAMPLE_001"
        result = _normalize_sample_id(sid)
        assert result == "SAMPLE_001"
        assert isinstance(result, str)

    def test_mixed_types_in_list_comprehension(self):
        raw_ids = [1, 2, 9007199254740993, "S001", 3.14]
        normalized = [_normalize_sample_id(sid) for sid in raw_ids]
        assert normalized[0] == 1
        assert normalized[1] == 2
        assert normalized[2] == 9007199254740993
        assert normalized[3] == "S001"
        assert isinstance(normalized[4], str)


class TestNormalizeSampleIdEdgeCases:
    """Edge cases for _normalize_sample_id."""

    def test_nan_float_raises(self):
        with pytest.raises(ValueError):
            _normalize_sample_id(np.nan)

    def test_zero(self):
        assert _normalize_sample_id(0) == 0
        assert _normalize_sample_id(0.0) == 0
        assert _normalize_sample_id(np.int64(0)) == 0
        assert _normalize_sample_id(np.float64(0.0)) == 0

    def test_negative_int(self):
        sid = -42
        result = _normalize_sample_id(sid)
        assert result == -42
        assert isinstance(result, int)

    def test_very_large_string_overflow(self):
        sid = "1234567890123456789012345678901234567890"
        result = _normalize_sample_id(sid)
        assert result == 1234567890123456789012345678901234567890
        assert isinstance(result, int)

    def test_float_with_large_fractional_part(self):
        sid = 0.0000000001
        result = _normalize_sample_id(sid)
        assert isinstance(result, str)
