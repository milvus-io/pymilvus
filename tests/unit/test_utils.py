import datetime
import struct
from datetime import timedelta

import pytest
from pymilvus.client import utils
from pymilvus.client.constants import LOGICAL_BITS, LOGICAL_BITS_MASK
from pymilvus.client.types import DataType
from pymilvus.client.utils import immutable_message_to_dict, replicate_checkpoint_to_dict
from pymilvus.exceptions import DescribeCollectionException, MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2


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

    def test_with_timedelta_whole_seconds(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.mkts_from_hybridts(hybridts, delta=timedelta(seconds=1))
        expected = (physical_ms + 1000) << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta_days(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        result = utils.mkts_from_hybridts(hybridts, delta=timedelta(days=1))
        expected = (physical_ms + 86400 * 1000) << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta_seconds_and_microseconds(self):
        physical_ms = 1000
        hybridts = physical_ms << LOGICAL_BITS
        delta = timedelta(seconds=1, microseconds=500000)  # 1500ms
        result = utils.mkts_from_hybridts(hybridts, delta=delta)
        expected = (physical_ms + 1500) << LOGICAL_BITS
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

    def test_with_timedelta_whole_seconds(self):
        result = utils.mkts_from_unixtime(1.0, delta=timedelta(seconds=1))
        expected = 2000 << LOGICAL_BITS
        assert result == expected

    def test_with_timedelta_seconds_and_microseconds(self):
        delta = timedelta(seconds=1, microseconds=500000)  # 1500ms
        result = utils.mkts_from_unixtime(1.0, delta=delta)
        expected = 2500 << LOGICAL_BITS
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

    def test_empty_values(self):
        entities = [{"type": DataType.BINARY_VECTOR, "values": []}]
        assert utils.check_invalid_binary_vector(entities) is False


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


class TestConvertStructFieldsToUserFormat:
    def test_nullable_struct_and_stored_sub_field_names(self):
        converted = utils.convert_struct_fields_to_user_format(
            [
                {
                    "field_id": 10,
                    "name": "metadata",
                    "description": "desc",
                    "nullable": True,
                    "fields": [
                        {
                            "field_id": 11,
                            "name": "metadata[score]",
                            "element_type": DataType.FLOAT,
                            "description": "score desc",
                            "params": {"max_capacity": "16", "mmap_enabled": True},
                        }
                    ],
                }
            ]
        )

        assert converted == [
            {
                "field_id": 10,
                "name": "metadata",
                "description": "desc",
                "type": DataType.ARRAY,
                "element_type": DataType.STRUCT,
                "params": {"max_capacity": "16"},
                "nullable": True,
                "struct_fields": [
                    {
                        "field_id": 11,
                        "name": "score",
                        "type": DataType.FLOAT,
                        "description": "score desc",
                        "params": {"mmap_enabled": True},
                    }
                ],
            }
        ]

    def test_strip_struct_sub_field_name_passthrough(self):
        assert utils.strip_struct_sub_field_name("metadata", "score") == "score"


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


class TestValidateIsoTimestamp:
    def test_valid_iso_timestamps(self):
        valid_timestamps = [
            "2023-01-01T00:00:00Z",
            "2023-01-01T12:30:45+00:00",
            "2023-01-01",
            "2023-01-01T00:00:00.123456Z",
        ]
        for ts in valid_timestamps:
            assert utils.validate_iso_timestamp(ts) is True, f"Expected {ts} to be valid"

    def test_invalid_iso_timestamps(self):
        invalid_timestamps = ["not-a-timestamp", "2023/01/01", "01-01-2023", ""]
        for ts in invalid_timestamps:
            assert utils.validate_iso_timestamp(ts) is False, f"Expected {ts} to be invalid"

    def test_invalid_type(self):
        assert utils.validate_iso_timestamp(None) is False
        assert utils.validate_iso_timestamp(123) is False


class TestReplicateCheckpointToDict:
    """Tests for replicate_checkpoint_to_dict helper."""

    def test_returns_none_for_none_input(self):
        assert replicate_checkpoint_to_dict(None) is None

    def test_returns_none_for_empty_proto(self):
        cp = common_pb2.ReplicateCheckpoint()
        assert replicate_checkpoint_to_dict(cp) is None

    def test_populated_with_message_id(self):
        cp = common_pb2.ReplicateCheckpoint(
            cluster_id="primary",
            pchannel="by-dev-rootcoord-dml_0",
            message_id=common_pb2.MessageID(id="msg-1", WAL_name=common_pb2.WALName.Pulsar),
            time_tick=12345,
        )
        result = replicate_checkpoint_to_dict(cp)
        assert result == {
            "cluster_id": "primary",
            "pchannel": "by-dev-rootcoord-dml_0",
            "message_id": {"id": "msg-1", "wal_name": "Pulsar"},
            "time_tick": 12345,
        }

    def test_populated_without_message_id(self):
        cp = common_pb2.ReplicateCheckpoint(
            cluster_id="primary",
            pchannel="ch0",
            time_tick=9,
        )
        result = replicate_checkpoint_to_dict(cp)
        assert result == {
            "cluster_id": "primary",
            "pchannel": "ch0",
            "message_id": None,
            "time_tick": 9,
        }

    def test_only_message_id_set(self):
        cp = common_pb2.ReplicateCheckpoint(
            message_id=common_pb2.MessageID(id="msg-1", WAL_name=common_pb2.WALName.Pulsar),
        )
        result = replicate_checkpoint_to_dict(cp)
        assert result == {
            "cluster_id": "",
            "pchannel": "",
            "message_id": {"id": "msg-1", "wal_name": "Pulsar"},
            "time_tick": 0,
        }


class TestImmutableMessageToDict:
    """Tests for immutable_message_to_dict."""

    def test_full_message(self):
        msg = common_pb2.ImmutableMessage(
            id=common_pb2.MessageID(id="msg-1", WAL_name=common_pb2.WALName.Pulsar),
            payload=b"\x01\x02",
            properties={"_t": "Insert"},
        )
        assert immutable_message_to_dict(msg) == {
            "message_id": {"id": "msg-1", "wal_name": "Pulsar"},
            "payload": b"\x01\x02",
            "properties": {"_t": "Insert"},
        }

    def test_unset_id_returns_none_message_id(self):
        msg = common_pb2.ImmutableMessage(payload=b"data")
        result = immutable_message_to_dict(msg)
        assert result["message_id"] is None
        assert result["payload"] == b"data"
        assert result["properties"] == {}

    def test_empty_message(self):
        msg = common_pb2.ImmutableMessage()
        assert immutable_message_to_dict(msg) == {
            "message_id": None,
            "payload": b"",
            "properties": {},
        }


class TestCheckStatusClassification:
    def test_input_error_surfaced(self):
        status = common_pb2.Status(
            code=1100,
            reason="invalid parameter",
            error_code=common_pb2.IllegalArgument,
            retriable=False,
            extra_info={"is_input_error": "true"},
        )
        with pytest.raises(MilvusException) as exc_info:
            utils.check_status(status)
        e = exc_info.value
        assert e.code == 1100
        assert e.is_input_error is True
        assert e.retriable is False

    def test_system_error_retriable(self):
        status = common_pb2.Status(
            code=2,
            reason="service unavailable",
            retriable=True,
        )
        with pytest.raises(MilvusException) as exc_info:
            utils.check_status(status)
        e = exc_info.value
        assert e.code == 2
        assert e.is_input_error is False
        assert e.retriable is True

    def test_defaults_when_unset(self):
        # A bare error status (no extra_info, no retriable) defaults to a
        # non-input, non-retriable system error.
        status = common_pb2.Status(code=500, reason="boom")
        with pytest.raises(MilvusException) as exc_info:
            utils.check_status(status)
        e = exc_info.value
        assert e.is_input_error is False
        assert e.retriable is False

    def test_from_status_preserves_subclass(self):
        status = common_pb2.Status(
            code=0,
            reason="can't find collection",
            error_code=common_pb2.CollectionNotExists,
            extra_info={"is_input_error": "true"},
        )
        e = DescribeCollectionException.from_status(status)
        assert isinstance(e, DescribeCollectionException)
        assert e.is_input_error is True
        assert e.compatible_code == common_pb2.CollectionNotExists
