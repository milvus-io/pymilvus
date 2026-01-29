# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for pymilvus/orm/utility.py timestamp functions."""

import time
from datetime import datetime, timedelta, timezone

import pytest
from pymilvus.client.constants import LOGICAL_BITS
from pymilvus.exceptions import MilvusException
from pymilvus.orm.utility import (
    hybridts_to_datetime,
    hybridts_to_unixtime,
    mkts_from_datetime,
    mkts_from_hybridts,
    mkts_from_unixtime,
)


class TestMktsFromHybridts:
    """Tests for mkts_from_hybridts function."""

    def test_mkts_from_hybridts_basic(self):
        """Test basic hybrid timestamp generation from existing hybrid timestamp."""
        base_ts = 1000 << LOGICAL_BITS
        result = mkts_from_hybridts(base_ts)
        assert isinstance(result, int)
        assert result == base_ts

    def test_mkts_from_hybridts_with_milliseconds(self):
        """Test hybrid timestamp generation with milliseconds offset."""
        base_ts = 1000 << LOGICAL_BITS
        result = mkts_from_hybridts(base_ts, milliseconds=1000.0)
        assert result > base_ts
        # The result should be shifted by 1000ms
        expected_physical = ((base_ts >> LOGICAL_BITS) + 1000) << LOGICAL_BITS
        assert result == expected_physical

    def test_mkts_from_hybridts_with_negative_milliseconds(self):
        """Test hybrid timestamp generation with negative milliseconds."""
        base_ts = 2000 << LOGICAL_BITS
        result = mkts_from_hybridts(base_ts, milliseconds=-500.0)
        assert result < base_ts

    def test_mkts_from_hybridts_with_delta(self):
        """Test hybrid timestamp generation with timedelta."""
        base_ts = 1000 << LOGICAL_BITS
        delta = timedelta(milliseconds=500)
        result = mkts_from_hybridts(base_ts, delta=delta)
        # timedelta uses microseconds, so 500ms = 500000 microseconds / 1000 = 500ms offset
        assert result > base_ts

    def test_mkts_from_hybridts_with_both_milliseconds_and_delta(self):
        """Test hybrid timestamp generation with both milliseconds and delta."""
        base_ts = 1000 << LOGICAL_BITS
        delta = timedelta(milliseconds=200)
        result = mkts_from_hybridts(base_ts, milliseconds=300.0, delta=delta)
        # Total offset should be 300 + 0.2 (from delta microseconds) ms
        assert result > base_ts

    def test_mkts_from_hybridts_preserves_logical_bits(self):
        """Test that logical bits are preserved during transformation."""
        logical_part = 12345
        physical_part = 1000
        base_ts = (physical_part << LOGICAL_BITS) | logical_part
        result = mkts_from_hybridts(base_ts, milliseconds=0.0)
        # Logical bits should be preserved
        assert (result & ((1 << LOGICAL_BITS) - 1)) == logical_part

    def test_mkts_from_hybridts_zero_input(self):
        """Test hybrid timestamp generation with zero input."""
        result = mkts_from_hybridts(0)
        assert result == 0

    def test_mkts_from_hybridts_large_value(self):
        """Test hybrid timestamp generation with large input value."""
        large_ts = (2**46) << LOGICAL_BITS  # Large but valid timestamp
        result = mkts_from_hybridts(large_ts)
        assert result == large_ts

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            pytest.param(
                {"hybridts": "not_an_int"},
                r"parameter hybridts should be type of int",
                id="invalid_hybridts_type",
            ),
            pytest.param(
                {"hybridts": 1000 << LOGICAL_BITS, "milliseconds": "invalid"},
                r"parameter milliseconds should be type of int or float",
                id="invalid_milliseconds_type",
            ),
            pytest.param(
                {"hybridts": 1000 << LOGICAL_BITS, "delta": "invalid"},
                r"parameter delta should be type of datetime\.timedelta",
                id="invalid_delta_type",
            ),
        ],
    )
    def test_mkts_from_hybridts_invalid_types(self, kwargs, error_match):
        """Test that invalid parameter types raise exceptions."""
        with pytest.raises(MilvusException, match=error_match):
            mkts_from_hybridts(**kwargs)


class TestMktsFromUnixtime:
    """Tests for mkts_from_unixtime function."""

    def test_mkts_from_unixtime_basic(self):
        """Test basic hybrid timestamp generation from Unix epoch."""
        epoch = 1609459200.0  # 2021-01-01 00:00:00 UTC
        result = mkts_from_unixtime(epoch)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_unixtime_current_time(self):
        """Test hybrid timestamp generation from current Unix time."""
        current_epoch = time.time()
        result = mkts_from_unixtime(current_epoch)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_unixtime_with_milliseconds(self):
        """Test hybrid timestamp generation with milliseconds offset."""
        epoch = 1609459200.0
        result_no_offset = mkts_from_unixtime(epoch)
        result_with_offset = mkts_from_unixtime(epoch, milliseconds=1000.0)
        assert result_with_offset > result_no_offset

    def test_mkts_from_unixtime_with_delta(self):
        """Test hybrid timestamp generation with timedelta."""
        epoch = 1609459200.0
        delta = timedelta(seconds=10)
        result_no_delta = mkts_from_unixtime(epoch)
        result_with_delta = mkts_from_unixtime(epoch, delta=delta)
        # Delta microseconds are converted to milliseconds
        assert result_with_delta >= result_no_delta

    def test_mkts_from_unixtime_integer_epoch(self):
        """Test hybrid timestamp generation with integer epoch."""
        epoch = 1609459200
        result = mkts_from_unixtime(epoch)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_unixtime_epoch_zero(self):
        """Test hybrid timestamp generation with epoch zero (Unix epoch start)."""
        result = mkts_from_unixtime(0.0)
        assert result == 0

    def test_mkts_from_unixtime_converts_correctly(self):
        """Test that conversion follows the expected formula."""
        epoch = 1000.0  # 1000 seconds since Unix epoch
        result = mkts_from_unixtime(epoch)
        # epoch in milliseconds = 1000 * 1000 = 1_000_000
        # Shifted by LOGICAL_BITS
        expected = (1000 * 1000) << LOGICAL_BITS
        assert result == expected

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            pytest.param(
                {"epoch": "not_a_number"},
                r"parameter epoch should be type of int or float",
                id="invalid_epoch_type",
            ),
            pytest.param(
                {"epoch": 1609459200.0, "milliseconds": "invalid"},
                r"parameter milliseconds should be type of int or float",
                id="invalid_milliseconds_type",
            ),
            pytest.param(
                {"epoch": 1609459200.0, "delta": "invalid"},
                r"parameter delta should be type of datetime\.timedelta",
                id="invalid_delta_type",
            ),
        ],
    )
    def test_mkts_from_unixtime_invalid_types(self, kwargs, error_match):
        """Test that invalid parameter types raise exceptions."""
        with pytest.raises(MilvusException, match=error_match):
            mkts_from_unixtime(**kwargs)


class TestMktsFromDatetime:
    """Tests for mkts_from_datetime function."""

    def test_mkts_from_datetime_basic(self):
        """Test basic hybrid timestamp generation from datetime."""
        dt = datetime(2021, 1, 1, 0, 0, 0)
        result = mkts_from_datetime(dt)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_datetime_current(self):
        """Test hybrid timestamp generation from current datetime."""
        dt = datetime.now()
        result = mkts_from_datetime(dt)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_datetime_with_milliseconds(self):
        """Test hybrid timestamp generation with milliseconds offset."""
        dt = datetime(2021, 1, 1, 0, 0, 0)
        result_no_offset = mkts_from_datetime(dt)
        result_with_offset = mkts_from_datetime(dt, milliseconds=1000.0)
        assert result_with_offset > result_no_offset

    def test_mkts_from_datetime_with_delta(self):
        """Test hybrid timestamp generation with timedelta."""
        dt = datetime(2021, 1, 1, 0, 0, 0)
        delta = timedelta(seconds=10)
        result_no_delta = mkts_from_datetime(dt)
        result_with_delta = mkts_from_datetime(dt, delta=delta)
        assert result_with_delta >= result_no_delta

    def test_mkts_from_datetime_with_timezone(self):
        """Test hybrid timestamp generation from timezone-aware datetime."""
        dt_utc = datetime(2021, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = mkts_from_datetime(dt_utc)
        assert isinstance(result, int)
        assert result > 0

    def test_mkts_from_datetime_with_microseconds(self):
        """Test hybrid timestamp generation from datetime with microseconds."""
        dt = datetime(2021, 6, 15, 12, 30, 45, 123456)
        result = mkts_from_datetime(dt)
        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.parametrize(
        "d_time",
        [
            pytest.param("not_a_datetime", id="string"),
            pytest.param(1609459200.0, id="float"),
            pytest.param(1609459200, id="int"),
        ],
    )
    def test_mkts_from_datetime_invalid_type(self, d_time):
        """Test that non-datetime input raises exception."""
        with pytest.raises(
            MilvusException, match=r"parameter d_time should be type of datetime\.datetime"
        ):
            mkts_from_datetime(d_time)


class TestHybridtsToDatetime:
    """Tests for hybridts_to_datetime function."""

    def test_hybridts_to_datetime_basic(self):
        """Test basic datetime conversion from hybrid timestamp."""
        dt = datetime(2021, 6, 15, 12, 0, 0)
        ts = mkts_from_datetime(dt)
        result = hybridts_to_datetime(ts)
        assert isinstance(result, datetime)

    def test_hybridts_to_datetime_roundtrip(self):
        """Test roundtrip conversion datetime -> hybridts -> datetime."""
        original_dt = datetime(2021, 6, 15, 12, 0, 0)
        ts = mkts_from_datetime(original_dt)
        result = hybridts_to_datetime(ts)
        # Allow some tolerance due to precision loss
        assert abs((result - original_dt).total_seconds()) < 1

    def test_hybridts_to_datetime_with_utc_timezone(self):
        """Test datetime conversion with UTC timezone."""
        ts = mkts_from_unixtime(1609459200.0)  # 2021-01-01 00:00:00 UTC
        result = hybridts_to_datetime(ts, tz=timezone.utc)
        assert result.tzinfo == timezone.utc
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1

    def test_hybridts_to_datetime_with_custom_timezone(self):
        """Test datetime conversion with custom timezone offset."""
        ts = mkts_from_unixtime(1609459200.0)
        # UTC+5:30 (e.g., India Standard Time)
        custom_tz = timezone(timedelta(hours=5, minutes=30))
        result = hybridts_to_datetime(ts, tz=custom_tz)
        assert result.tzinfo == custom_tz

    def test_hybridts_to_datetime_without_timezone(self):
        """Test datetime conversion without timezone (local time)."""
        ts = mkts_from_unixtime(1609459200.0)
        result = hybridts_to_datetime(ts)
        # Should be naive datetime (no tzinfo)
        assert result.tzinfo is None

    @pytest.mark.parametrize(
        "invalid_tz",
        [
            pytest.param("invalid", id="string"),
            pytest.param(5, id="integer"),
            pytest.param(timedelta(hours=5), id="timedelta"),
        ],
    )
    def test_hybridts_to_datetime_invalid_timezone(self, invalid_tz):
        """Test that invalid timezone type raises exception."""
        ts = mkts_from_unixtime(1609459200.0)
        with pytest.raises(
            MilvusException, match=r"parameter tz should be type of datetime\.timezone"
        ):
            hybridts_to_datetime(ts, tz=invalid_tz)

    def test_hybridts_to_datetime_from_zero(self):
        """Test datetime conversion from zero hybrid timestamp."""
        result = hybridts_to_datetime(0, tz=timezone.utc)
        assert result.year == 1970
        assert result.month == 1
        assert result.day == 1


class TestHybridtsToUnixtime:
    """Tests for hybridts_to_unixtime function."""

    def test_hybridts_to_unixtime_basic(self):
        """Test basic Unix time conversion from hybrid timestamp."""
        epoch = 1609459200.0
        ts = mkts_from_unixtime(epoch)
        result = hybridts_to_unixtime(ts)
        assert isinstance(result, float)
        assert abs(result - epoch) < 0.001  # Allow small precision difference

    def test_hybridts_to_unixtime_roundtrip(self):
        """Test roundtrip conversion epoch -> hybridts -> epoch."""
        original_epoch = 1609459200.0
        ts = mkts_from_unixtime(original_epoch)
        result = hybridts_to_unixtime(ts)
        assert abs(result - original_epoch) < 0.001

    def test_hybridts_to_unixtime_from_zero(self):
        """Test Unix time conversion from zero hybrid timestamp."""
        result = hybridts_to_unixtime(0)
        assert result == 0.0

    def test_hybridts_to_unixtime_ignores_logical_bits(self):
        """Test that logical bits are ignored in conversion."""
        physical_ms = 1000000  # 1000 seconds in milliseconds
        logical_part = 12345
        ts = (physical_ms << LOGICAL_BITS) | logical_part
        result = hybridts_to_unixtime(ts)
        # Should return 1000.0 seconds regardless of logical part
        assert result == 1000.0


class TestTimestampIntegration:
    """Integration tests for timestamp conversion functions."""

    def test_datetime_to_unixtime_chain(self):
        """Test chaining datetime -> hybridts -> unixtime."""
        dt = datetime(2021, 6, 15, 12, 0, 0)
        expected_epoch = dt.timestamp()
        ts = mkts_from_datetime(dt)
        result_epoch = hybridts_to_unixtime(ts)
        # Allow for floating point precision issues
        assert abs(result_epoch - expected_epoch) < 0.001

    def test_unixtime_to_datetime_chain(self):
        """Test chaining unixtime -> hybridts -> datetime."""
        epoch = 1623758400.0  # 2021-06-15 12:00:00 UTC
        ts = mkts_from_unixtime(epoch)
        result_dt = hybridts_to_datetime(ts, tz=timezone.utc)
        assert result_dt.year == 2021
        assert result_dt.month == 6
        assert result_dt.day == 15
        assert result_dt.hour == 12

    def test_hybridts_modifications_chain(self):
        """Test modifying hybrid timestamp and converting back."""
        original_epoch = 1609459200.0
        ts = mkts_from_unixtime(original_epoch)

        # Add 1 second (1000ms)
        modified_ts = mkts_from_hybridts(ts, milliseconds=1000.0)
        result_epoch = hybridts_to_unixtime(modified_ts)

        assert abs(result_epoch - (original_epoch + 1.0)) < 0.001

    def test_current_time_consistency(self):
        """Test that current time conversions are consistent."""
        now = datetime.now()
        epoch_now = time.time()

        ts_from_dt = mkts_from_datetime(now)
        ts_from_epoch = mkts_from_unixtime(epoch_now)

        # Should be very close (within 1 second)
        diff = abs(ts_from_dt - ts_from_epoch)
        # Difference should be less than 1 second worth of hybrid timestamp
        assert diff < (1000 << LOGICAL_BITS)

    def test_various_date_ranges(self):
        """Test timestamp functions with various date ranges."""
        test_dates = [
            datetime(2000, 1, 1, 0, 0, 0),
            datetime(2010, 6, 15, 12, 30, 45),
            datetime(2020, 12, 31, 23, 59, 59),
            datetime(2025, 7, 4, 8, 0, 0),
        ]

        for dt in test_dates:
            ts = mkts_from_datetime(dt)
            assert isinstance(ts, int)
            assert ts > 0

            recovered_dt = hybridts_to_datetime(ts)
            # Allow 1 second tolerance
            assert abs((recovered_dt - dt).total_seconds()) < 1
