import datetime
import struct
from datetime import timedelta

import pytest
from pymilvus.client import utils
from pymilvus.client.constants import LOGICAL_BITS, LOGICAL_BITS_MASK
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ParamError


class TestGetServerType:
    def test_get_server_type(self):
        urls_and_wants = [
            ("in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com", "zilliz"),
            ("something.abc.com", "milvus"),
            ("something.zillizcloud.cn", "zilliz"),
        ]
        for url, want in urls_and_wants:
            assert utils.get_server_type(url) == want

    def test_get_server_type_case_insensitive(self):
        assert utils.get_server_type("ZILLIZ.example.com") == "zilliz"
        assert utils.get_server_type("ZiLlIzCloud.com") == "zilliz"

    def test_get_server_type_non_string(self):
        assert utils.get_server_type(None) == "milvus"
        assert utils.get_server_type(123) == "milvus"


class TestHybridtsToUnixtime:
    def test_basic_conversion(self):
        # Create a hybrid timestamp with known physical time (1000ms) and logical part (0)
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.hybridts_to_unixtime(hybridts)
        assert result == 1.0  # 1000ms = 1.0 seconds

    def test_with_logical_part(self):
        # The logical part should not affect the unix time conversion
        physical_ms = 5000
        logical = 100
        hybridts = (physical_ms << LOGICAL_BITS) | logical
        result = utils.hybridts_to_unixtime(hybridts)
        assert result == 5.0  # 5000ms = 5.0 seconds

    def test_zero_timestamp(self):
        result = utils.hybridts_to_unixtime(0)
        assert result == 0.0

    def test_large_timestamp(self):
        # Test with a realistic timestamp (e.g., around 2023)
        # Unix timestamp for 2023-01-01 00:00:00 UTC is 1672531200
        physical_ms = 1672531200000  # Convert to milliseconds
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.hybridts_to_unixtime(hybridts)
        assert result == 1672531200.0


class TestMktsFromHybridts:
    def test_basic_conversion(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.mkts_from_hybridts(hybridts)
        assert result == hybridts

    def test_with_milliseconds_offset(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        offset_ms = 500.0
        result = utils.mkts_from_hybridts(hybridts, milliseconds=offset_ms)
        expected = (physical_ms + int(offset_ms)) << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        delta = timedelta(microseconds=500000)  # 500ms
        result = utils.mkts_from_hybridts(hybridts, delta=delta)
        # 500000 microseconds = 500ms
        expected = (physical_ms + 500) << LOGICAL_BITS
        assert result == expected

    def test_with_both_milliseconds_and_delta(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        offset_ms = 100.0
        delta = timedelta(microseconds=200000)  # 200ms
        result = utils.mkts_from_hybridts(hybridts, milliseconds=offset_ms, delta=delta)
        # Total offset: 100 + 200 = 300ms
        expected = (physical_ms + 300) << LOGICAL_BITS
        assert result == expected

    def test_preserves_logical_part(self):
        physical_ms = 1000
        logical = 42
        hybridts = (physical_ms << LOGICAL_BITS) | logical
        result = utils.mkts_from_hybridts(hybridts)
        assert result & LOGICAL_BITS_MASK == logical

    def test_invalid_milliseconds_type(self):
        hybridts = 1000 << LOGICAL_BITS
        with pytest.raises(
            MilvusException, match="parameter milliseconds should be type of int or float"
        ):
            utils.mkts_from_hybridts(hybridts, milliseconds="invalid")

    def test_invalid_delta_type(self):
        hybridts = 1000 << LOGICAL_BITS
        with pytest.raises(
            MilvusException, match=r"parameter delta should be type of datetime\.timedelta"
        ):
            utils.mkts_from_hybridts(hybridts, delta="invalid")

    def test_invalid_hybridts_type(self):
        with pytest.raises(MilvusException, match="parameter hybridts should be type of int"):
            utils.mkts_from_hybridts("invalid")


class TestMktsFromUnixtime:
    def test_basic_conversion(self):
        epoch = 1.0  # 1 second = 1000ms
        result = utils.mkts_from_unixtime(epoch)
        expected = 1000 << LOGICAL_BITS
        assert result == expected

    def test_with_milliseconds_offset(self):
        epoch = 1.0
        offset_ms = 500.0
        result = utils.mkts_from_unixtime(epoch, milliseconds=offset_ms)
        # 1.0 second + 0.5 seconds = 1.5 seconds = 1500ms
        expected = 1500 << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta(self):
        epoch = 1.0
        delta = timedelta(microseconds=500000)  # 500ms
        result = utils.mkts_from_unixtime(epoch, delta=delta)
        expected = 1500 << LOGICAL_BITS
        assert result == expected

    def test_with_both_milliseconds_and_delta(self):
        epoch = 1.0
        offset_ms = 100.0
        delta = timedelta(microseconds=200000)  # 200ms
        result = utils.mkts_from_unixtime(epoch, milliseconds=offset_ms, delta=delta)
        # 1000 + 100 + 200 = 1300ms
        expected = 1300 << LOGICAL_BITS
        assert result == expected

    def test_zero_epoch(self):
        result = utils.mkts_from_unixtime(0)
        assert result == 0

    def test_float_epoch(self):
        epoch = 1.5  # 1500ms
        result = utils.mkts_from_unixtime(epoch)
        expected = 1500 << LOGICAL_BITS
        assert result == expected

    def test_invalid_epoch_type(self):
        with pytest.raises(MilvusException, match="parameter epoch should be type of int or float"):
            utils.mkts_from_unixtime("invalid")

    def test_invalid_milliseconds_type(self):
        with pytest.raises(
            MilvusException, match="parameter milliseconds should be type of int or float"
        ):
            utils.mkts_from_unixtime(1.0, milliseconds="invalid")

    def test_invalid_delta_type(self):
        with pytest.raises(
            MilvusException, match=r"parameter delta should be type of datetime\.timedelta"
        ):
            utils.mkts_from_unixtime(1.0, delta="invalid")


class TestMktsFromDatetime:
    def test_basic_conversion(self):
        # Use a known datetime
        dt = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        result = utils.mkts_from_datetime(dt)
        # Verify it converts correctly by checking round-trip
        expected_epoch = dt.timestamp()
        expected = int(expected_epoch * 1000) << LOGICAL_BITS
        assert result == expected

    def test_with_milliseconds_offset(self):
        dt = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        offset_ms = 500.0
        result = utils.mkts_from_datetime(dt, milliseconds=offset_ms)
        expected_epoch = dt.timestamp() + 0.5  # 500ms = 0.5s
        expected = int(expected_epoch * 1000) << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta(self):
        dt = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        delta = timedelta(microseconds=500000)  # 500ms
        result = utils.mkts_from_datetime(dt, delta=delta)
        expected_epoch = dt.timestamp() + 0.5
        expected = int(expected_epoch * 1000) << LOGICAL_BITS
        assert result == expected

    def test_invalid_datetime_type(self):
        with pytest.raises(
            MilvusException, match=r"parameter d_time should be type of datetime\.datetime"
        ):
            utils.mkts_from_datetime("2023-01-01")

    def test_invalid_datetime_type_int(self):
        with pytest.raises(
            MilvusException, match=r"parameter d_time should be type of datetime\.datetime"
        ):
            utils.mkts_from_datetime(1672531200)


class TestCurrentTimeMs:
    def test_returns_string(self):
        result = utils.current_time_ms()
        assert isinstance(result, str)

    def test_returns_reasonable_value(self):
        result = utils.current_time_ms()
        # Should be parseable as an integer
        value = int(result)
        # Should be a reasonable timestamp (after year 2020)
        assert value > 1577836800000  # Jan 1, 2020 in ms


class TestDumps:
    def test_dumps_dict(self):
        data = {"key": "value", "number": 42}
        result = utils.dumps(data)
        assert '"key"' in result
        assert '"value"' in result
        assert "42" in result

    def test_dumps_string(self):
        data = "hello"
        result = utils.dumps(data)
        assert result == "hello"

    def test_dumps_number(self):
        result = utils.dumps(42)
        assert result == "42"

    def test_dumps_boolean(self):
        result = utils.dumps(True)
        assert result == "True"


class TestCheckInvalidBinaryVector:
    def test_valid_binary_vectors(self):
        entities = [
            {
                "type": DataType.BINARY_VECTOR,
                "values": [b"\x00\x01", b"\x02\x03"],
            }
        ]
        assert utils.check_invalid_binary_vector(entities) is True

    def test_empty_entities(self):
        assert utils.check_invalid_binary_vector([]) is True

    def test_non_binary_vector_entities(self):
        entities = [
            {
                "type": DataType.FLOAT_VECTOR,
                "values": [[0.1, 0.2], [0.3, 0.4]],
            }
        ]
        assert utils.check_invalid_binary_vector(entities) is True

    def test_inconsistent_dimensions(self):
        entities = [
            {
                "type": DataType.BINARY_VECTOR,
                "values": [b"\x00\x01", b"\x02\x03\x04"],  # Different lengths
            }
        ]
        assert utils.check_invalid_binary_vector(entities) is False

    def test_non_bytes_values(self):
        entities = [
            {
                "type": DataType.BINARY_VECTOR,
                "values": [[0, 1], [2, 3]],  # Lists instead of bytes
            }
        ]
        assert utils.check_invalid_binary_vector(entities) is False


class TestVectorTypeChecks:
    def test_is_sparse_vector_type(self):
        assert utils.is_sparse_vector_type(DataType.SPARSE_FLOAT_VECTOR) is True
        assert utils.is_sparse_vector_type(DataType.FLOAT_VECTOR) is False
        assert utils.is_sparse_vector_type(DataType.BINARY_VECTOR) is False

    def test_is_dense_float_vector_type(self):
        assert utils.is_dense_float_vector_type(DataType.FLOAT_VECTOR) is True
        assert utils.is_dense_float_vector_type(DataType.FLOAT16_VECTOR) is True
        assert utils.is_dense_float_vector_type(DataType.BFLOAT16_VECTOR) is True
        assert utils.is_dense_float_vector_type(DataType.SPARSE_FLOAT_VECTOR) is False
        assert utils.is_dense_float_vector_type(DataType.BINARY_VECTOR) is False
        assert utils.is_dense_float_vector_type(DataType.INT8_VECTOR) is False

    def test_is_float_vector_type(self):
        assert utils.is_float_vector_type(DataType.FLOAT_VECTOR) is True
        assert utils.is_float_vector_type(DataType.FLOAT16_VECTOR) is True
        assert utils.is_float_vector_type(DataType.BFLOAT16_VECTOR) is True
        assert utils.is_float_vector_type(DataType.SPARSE_FLOAT_VECTOR) is True
        assert utils.is_float_vector_type(DataType.BINARY_VECTOR) is False
        assert utils.is_float_vector_type(DataType.INT8_VECTOR) is False

    def test_is_binary_vector_type(self):
        assert utils.is_binary_vector_type(DataType.BINARY_VECTOR) is True
        assert utils.is_binary_vector_type(DataType.FLOAT_VECTOR) is False

    def test_is_int_vector_type(self):
        assert utils.is_int_vector_type(DataType.INT8_VECTOR) is True
        assert utils.is_int_vector_type(DataType.FLOAT_VECTOR) is False

    def test_is_vector_type(self):
        assert utils.is_vector_type(DataType.FLOAT_VECTOR) is True
        assert utils.is_vector_type(DataType.FLOAT16_VECTOR) is True
        assert utils.is_vector_type(DataType.BFLOAT16_VECTOR) is True
        assert utils.is_vector_type(DataType.SPARSE_FLOAT_VECTOR) is True
        assert utils.is_vector_type(DataType.BINARY_VECTOR) is True
        assert utils.is_vector_type(DataType.INT8_VECTOR) is True
        assert utils.is_vector_type(DataType.INT64) is False
        assert utils.is_vector_type(DataType.VARCHAR) is False


class TestSparseParseSingleRow:
    def test_basic_parsing(self):
        # Create sparse vector data: {0: 1.0, 1: 2.0}
        data = struct.pack("If", 0, 1.0) + struct.pack("If", 1, 2.0)
        result = utils.sparse_parse_single_row(data)
        assert result == {0: 1.0, 1: 2.0}

    def test_empty_data(self):
        result = utils.sparse_parse_single_row(b"")
        assert result == {}

    def test_invalid_length(self):
        # Data length must be multiple of 8
        with pytest.raises(ParamError, match="length of data must be a multiple of 8"):
            utils.sparse_parse_single_row(b"\x00\x01\x02")


class TestTraverseInfo:
    def test_basic_traverse(self):
        fields_info = [
            {"name": "id", "is_primary": True},
            {"name": "vector", "is_primary": False},
            {"name": "text", "is_primary": False},
        ]
        location, primary_key_loc, auto_id_loc = utils.traverse_info(fields_info)
        assert location == {"id": 0, "vector": 1, "text": 2}
        assert primary_key_loc == 0
        assert auto_id_loc is None

    def test_with_auto_id(self):
        fields_info = [
            {"name": "id", "is_primary": True, "auto_id": True},
            {"name": "vector", "is_primary": False},
        ]
        location, primary_key_loc, auto_id_loc = utils.traverse_info(fields_info)
        # Auto ID field should not be in location
        assert "id" not in location
        assert location == {"vector": 1}
        assert primary_key_loc == 0
        assert auto_id_loc == 0


class TestTraverseUpsertInfo:
    def test_basic_traverse(self):
        fields_info = [
            {"name": "id", "is_primary": True},
            {"name": "vector", "is_primary": False},
        ]
        location, primary_key_loc = utils.traverse_upsert_info(fields_info)
        assert location == {"id": 0, "vector": 1}
        assert primary_key_loc == 0

    def test_with_auto_id(self):
        # For upsert, auto_id field should still be in location
        fields_info = [
            {"name": "id", "is_primary": True, "auto_id": True},
            {"name": "vector", "is_primary": False},
        ]
        location, primary_key_loc = utils.traverse_upsert_info(fields_info)
        assert location == {"id": 0, "vector": 1}
        assert primary_key_loc == 0


class TestGetParams:
    def test_basic_params(self):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        result = utils.get_params(search_params)
        assert result == {"metric_type": "L2", "nprobe": 10}

    def test_flat_params(self):
        search_params = {"metric_type": "L2", "nprobe": 10}
        result = utils.get_params(search_params)
        assert result == {"metric_type": "L2", "nprobe": 10}

    def test_no_params_key(self):
        search_params = {"metric_type": "IP", "ef": 64}
        result = utils.get_params(search_params)
        assert result == {"metric_type": "IP", "ef": 64}

    def test_ambiguous_parameter_error(self):
        search_params = {"nprobe": 20, "params": {"nprobe": 10}}
        with pytest.raises(ParamError, match="ambiguous parameter"):
            utils.get_params(search_params)

    def test_same_value_no_error(self):
        # If the value is the same, no error should be raised
        search_params = {"nprobe": 10, "params": {"nprobe": 10}}
        result = utils.get_params(search_params)
        assert result["nprobe"] == 10


class TestUtils:
    """Legacy test class kept for compatibility."""

    def test_get_server_type(self):
        urls_and_wants = [
            ("in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com", "zilliz"),
            ("something.abc.com", "milvus"),
            ("something.zillizcloud.cn", "zilliz"),
        ]
        for url, want in urls_and_wants:
            assert utils.get_server_type(url) == want
