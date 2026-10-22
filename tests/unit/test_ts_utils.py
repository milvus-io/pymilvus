from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pymilvus.client import ts_utils
from pymilvus.client.cache import CollectionTsCache, GlobalCache
from pymilvus.exceptions import InvalidConsistencyLevel
from pymilvus.grpc_gen.common_pb2 import ConsistencyLevel


@pytest.fixture(autouse=True)
def isolated_collection_timestamps(monkeypatch):
    monkeypatch.setattr(GlobalCache, "collection_ts", CollectionTsCache())


class TestTsUtils:

    def test_update_and_get(self):
        # Verify update_collection_ts and get_collection_ts
        collection_name = "coll1"
        endpoint = "localhost:19530"
        db_name = "default"

        # Initial state should be 0
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 0

        # Update and verify
        ts_utils.update_collection_ts(collection_name, 100, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 100

        # Update with smaller timestamp should be ignored
        ts_utils.update_collection_ts(collection_name, 50, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 100

        # Update with larger timestamp should succeed
        ts_utils.update_collection_ts(collection_name, 200, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 200

        # Test with empty endpoint/db (default behavior)
        ts_utils.update_collection_ts(collection_name, 300)
        assert ts_utils.get_collection_ts(collection_name) == 300
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 200

    def test_get_current_bounded_ts(self):
        ts = ts_utils.get_bounded_ts()
        assert ts == 2

    def test_get_eventually_ts(self):
        ts = ts_utils.get_eventually_ts()
        assert ts == 1

    @pytest.mark.parametrize(
        "other_collection,other_endpoint,other_db",
        [
            pytest.param("coll2", "ep1", "db1", id="collection"),
            pytest.param("coll1", "ep2", "db1", id="endpoint"),
            pytest.param("coll1", "ep1", "db2", id="database"),
        ],
    )
    def test_mutation_callback_keeps_scopes_separate(
        self, other_collection, other_endpoint, other_db
    ):
        update = ts_utils.update_ts_on_mutation("coll1", "ep1", "db1")
        update_other = ts_utils.update_ts_on_mutation(other_collection, other_endpoint, other_db)

        update(SimpleNamespace(timestamp=2000))
        assert ts_utils.get_collection_ts(other_collection, other_endpoint, other_db) == 0
        request_args = {"consistency_level": "Session"}
        assert (
            ts_utils.construct_guarantee_ts(
                other_collection, request_args, other_endpoint, other_db
            )
            is False
        )
        assert request_args["guarantee_timestamp"] == 1

        update_other(SimpleNamespace(timestamp=3000))
        update(SimpleNamespace(timestamp=1000))
        assert ts_utils.get_collection_ts("coll1", "ep1", "db1") == 2000
        assert ts_utils.get_collection_ts(other_collection, other_endpoint, other_db) == 3000

        update(SimpleNamespace(timestamp=4000))
        request_args = {"consistency_level": "Session"}
        assert ts_utils.construct_guarantee_ts("coll1", request_args, "ep1", "db1") is False
        assert request_args["guarantee_timestamp"] == 4000
        assert ts_utils.get_collection_ts(other_collection, other_endpoint, other_db) == 3000

    @pytest.mark.parametrize(
        "timestamp,tz,expected_seconds,expected_datetime",
        [
            pytest.param(
                0, timezone.utc, 0.0, datetime(1970, 1, 1, tzinfo=timezone.utc), id="epoch"
            ),
            pytest.param(
                327680009,
                timezone.utc,
                1.25,
                datetime(1970, 1, 1, 0, 0, 1, 250000, tzinfo=timezone.utc),
                id="milliseconds-and-logical-counter",
            ),
            pytest.param(
                327680009,
                timezone(timedelta(hours=8)),
                1.25,
                datetime(1970, 1, 1, 8, 0, 1, 250000, tzinfo=timezone(timedelta(hours=8))),
                id="explicit-timezone",
            ),
        ],
    )
    def test_collection_time_conversion(self, timestamp, tz, expected_seconds, expected_datetime):
        # Hybrid timestamp 327680009 contains 1250 physical milliseconds and logical counter 9.
        ts_utils.update_collection_ts("coll1", timestamp, "ep1", "db1")

        assert ts_utils.get_collection_timestamp("coll1", "ep1", "db1") == expected_seconds
        actual_datetime = ts_utils.get_collection_datetime("coll1", "ep1", "db1", tz=tz)
        assert actual_datetime == expected_datetime
        assert actual_datetime.tzinfo == tz


@pytest.mark.parametrize(
    "kwargs,endpoint,expected,use_default",
    [
        pytest.param({"iterator": True}, "ep1", {"iterator": True}, True, id="iterator"),
        pytest.param(
            {"iterator": True, "guarantee_timestamp": 17},
            "ep1",
            {"iterator": True, "guarantee_timestamp": 17},
            True,
            id="iterator-keeps-ts",
        ),
        pytest.param({}, "ep1", {"guarantee_timestamp": 2000}, True, id="default-cached"),
        pytest.param({}, "ep2", {"guarantee_timestamp": 1}, True, id="default-other-endpoint"),
        pytest.param(
            {"consistency_level": "Strong"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Strong, "guarantee_timestamp": 0},
            False,
            id="strong",
        ),
        pytest.param(
            {"consistency_level": ConsistencyLevel.Strong, "guarantee_timestamp": 123},
            "ep1",
            {"consistency_level": ConsistencyLevel.Strong, "guarantee_timestamp": 0},
            False,
            id="strong-overrides-ts",
        ),
        pytest.param(
            {"consistency_level": "Bounded"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Bounded, "guarantee_timestamp": 2},
            False,
            id="bounded",
        ),
        pytest.param(
            {"consistency_level": ConsistencyLevel.Bounded, "guarantee_timestamp": 123},
            "ep1",
            {"consistency_level": ConsistencyLevel.Bounded, "guarantee_timestamp": 2},
            False,
            id="bounded-overrides-ts",
        ),
        pytest.param(
            {"consistency_level": "Session"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Session, "guarantee_timestamp": 2000},
            False,
            id="session-cached",
        ),
        pytest.param(
            {"consistency_level": ConsistencyLevel.Session, "guarantee_timestamp": 123},
            "ep1",
            {"consistency_level": ConsistencyLevel.Session, "guarantee_timestamp": 2000},
            False,
            id="session-overrides-ts",
        ),
        pytest.param(
            {"consistency_level": "Session"},
            "ep2",
            {"consistency_level": ConsistencyLevel.Session, "guarantee_timestamp": 1},
            False,
            id="session-other-endpoint",
        ),
        pytest.param(
            {"consistency_level": "Eventually"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Eventually, "guarantee_timestamp": 1},
            False,
            id="eventually",
        ),
        pytest.param(
            {"consistency_level": "Customized", "guarantee_timestamp": 123},
            "ep1",
            {"consistency_level": ConsistencyLevel.Customized, "guarantee_timestamp": 123},
            False,
            id="customized-keeps-ts",
        ),
        pytest.param(
            {"consistency_level": "Customized", "guarantee_timestamp": 0},
            "ep1",
            {"consistency_level": ConsistencyLevel.Customized, "guarantee_timestamp": 0},
            False,
            id="customized-keeps-zero-ts",
        ),
        pytest.param(
            {"consistency_level": ConsistencyLevel.Customized},
            "ep1",
            {"consistency_level": ConsistencyLevel.Customized, "guarantee_timestamp": 1},
            False,
            id="customized-default-ts",
        ),
    ],
)
def test_construct_guarantee_ts(kwargs, endpoint, expected, use_default):
    ts_utils.update_collection_ts("coll1", 2000, "ep1", "db1")
    request_args = dict(kwargs)

    result = ts_utils.construct_guarantee_ts("coll1", request_args, endpoint, "db1")

    assert result is use_default
    assert request_args == expected


@pytest.mark.parametrize("level", ["invalid", 999, 1.5], ids=["name", "enum", "type"])
def test_construct_guarantee_ts_rejects_invalid_consistency(level):
    with pytest.raises(InvalidConsistencyLevel, match="invalid consistency level"):
        ts_utils.construct_guarantee_ts("coll1", {"consistency_level": level}, "ep1", "db1")
