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
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.hybridts_to_unixtime(hybridts)
        assert result == 1.0

    def test_with_logical_part(self):
        physical_ms = 5000
        logical = 100
        hybridts = (physical_ms << LOGICAL_BITS) | logical
        result = utils.hybridts_to_unixtime(hybridts)
        assert result == 5.0

    def test_zero_timestamp(self):
        result = utils.hybridts_to_unixtime(0)
        assert result == 0.0

    def test_large_timestamp(self):
        physical_ms = 1672531200000
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
        expected = (physical_ms + 500) << LOGICAL_BITS
        assert result == expected

    def test_with_both_milliseconds_and_delta(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        offset_ms = 100.0
        delta = timedelta(microseconds=200000)  # 200ms
        result = utils.mkts_from_hybridts(hybridts, milliseconds=offset_ms, delta=delta)
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
        epoch = 1.0
        result = utils.mkts_from_unixtime(epoch)
        expected = 1000 << LOGICAL_BITS
        assert result == expected

    def test_with_milliseconds_offset(self):
        epoch = 1.0
        offset_ms = 500.0
        result = utils.mkts_from_unixtime(epoch, milliseconds=offset_ms)
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
        expected = 1300 << LOGICAL_BITS
        assert result == expected

    def test_zero_epoch(self):
        result = utils.mkts_from_unixtime(0)
        assert result == 0

    def test_float_epoch(self):
        epoch = 1.5
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
        dt = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        result = utils.mkts_from_datetime(dt)
        expected_epoch = dt.timestamp()
        expected = int(expected_epoch * 1000) << LOGICAL_BITS
        assert result == expected

    def test_with_milliseconds_offset(self):
        dt = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        offset_ms = 500.0
        result = utils.mkts_from_datetime(dt, milliseconds=offset_ms)
        expected_epoch = dt.timestamp() + 0.5
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
        value = int(result)
        assert value > 1577836800000  # Jan 1, 2020 in ms


class TestDumps:
    def test_dumps_dict(self):
        data = {"key": "value", "number": 42}
        result = utils.dumps(data)
        assert '"key"' in result
        assert '"value"' in result
        assert "42" in result

    def test_dumps_string(self):
        assert utils.dumps("hello") == "hello"

    def test_dumps_number(self):
        assert utils.dumps(42) == "42"

    def test_dumps_boolean(self):
        assert utils.dumps(True) == "True"


class TestCheckInvalidBinaryVector:
    def test_valid_binary_vectors(self):
        entities = [{"type": DataType.BINARY_VECTOR, "values": [b"\x00\x01", b"\x02\x03"]}]
        assert utils.check_invalid_binary_vector(entities) is True

    def test_empty_entities(self):
        assert utils.check_invalid_binary_vector([]) is True

    def test_non_binary_vector_entities(self):
        entities = [{"type": DataType.FLOAT_VECTOR, "values": [[0.1, 0.2], [0.3, 0.4]]}]
        assert utils.check_invalid_binary_vector(entities) is True

    def test_inconsistent_dimensions(self):
        entities = [{"type": DataType.BINARY_VECTOR, "values": [b"\x00\x01", b"\x02\x03\x04"]}]
        assert utils.check_invalid_binary_vector(entities) is False

    def test_non_bytes_values(self):
        entities = [{"type": DataType.BINARY_VECTOR, "values": [[0, 1], [2, 3]]}]
        assert utils.check_invalid_binary_vector(entities) is False


# ── TestVectorTypeChecks (parametrized) ───────────────────────────────────────


@pytest.mark.parametrize(
    "dtype,fn_name,expected",
    [
        # is_sparse_vector_type
        (DataType.SPARSE_FLOAT_VECTOR, "is_sparse_vector_type", True),
        (DataType.FLOAT_VECTOR, "is_sparse_vector_type", False),
        (DataType.BINARY_VECTOR, "is_sparse_vector_type", False),
        # is_dense_float_vector_type
        (DataType.FLOAT_VECTOR, "is_dense_float_vector_type", True),
        (DataType.FLOAT16_VECTOR, "is_dense_float_vector_type", True),
        (DataType.BFLOAT16_VECTOR, "is_dense_float_vector_type", True),
        (DataType.SPARSE_FLOAT_VECTOR, "is_dense_float_vector_type", False),
        (DataType.BINARY_VECTOR, "is_dense_float_vector_type", False),
        (DataType.INT8_VECTOR, "is_dense_float_vector_type", False),
        # is_float_vector_type
        (DataType.FLOAT_VECTOR, "is_float_vector_type", True),
        (DataType.FLOAT16_VECTOR, "is_float_vector_type", True),
        (DataType.BFLOAT16_VECTOR, "is_float_vector_type", True),
        (DataType.SPARSE_FLOAT_VECTOR, "is_float_vector_type", True),
        (DataType.BINARY_VECTOR, "is_float_vector_type", False),
        (DataType.INT8_VECTOR, "is_float_vector_type", False),
        # is_binary_vector_type
        (DataType.BINARY_VECTOR, "is_binary_vector_type", True),
        (DataType.FLOAT_VECTOR, "is_binary_vector_type", False),
        # is_int_vector_type
        (DataType.INT8_VECTOR, "is_int_vector_type", True),
        (DataType.FLOAT_VECTOR, "is_int_vector_type", False),
        # is_vector_type
        (DataType.FLOAT_VECTOR, "is_vector_type", True),
        (DataType.FLOAT16_VECTOR, "is_vector_type", True),
        (DataType.BFLOAT16_VECTOR, "is_vector_type", True),
        (DataType.SPARSE_FLOAT_VECTOR, "is_vector_type", True),
        (DataType.BINARY_VECTOR, "is_vector_type", True),
        (DataType.INT8_VECTOR, "is_vector_type", True),
        (DataType.INT64, "is_vector_type", False),
        (DataType.VARCHAR, "is_vector_type", False),
    ],
)
def test_vector_type_checks(dtype, fn_name, expected):
    assert getattr(utils, fn_name)(dtype) is expected


class TestSparseParseSingleRow:
    def test_basic_parsing(self):
        data = struct.pack("If", 0, 1.0) + struct.pack("If", 1, 2.0)
        result = utils.sparse_parse_single_row(data)
        assert result == {0: 1.0, 1: 2.0}

    def test_empty_data(self):
        result = utils.sparse_parse_single_row(b"")
        assert result == {}

    def test_invalid_length(self):
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
