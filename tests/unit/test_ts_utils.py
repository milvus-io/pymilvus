import pytest
from pymilvus.client import ts_utils
from pymilvus.client.cache import CollectionTsCache, GlobalCache
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
            {"consistency_level": "Bounded"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Bounded, "guarantee_timestamp": 2},
            False,
            id="bounded",
        ),
        pytest.param(
            {"consistency_level": "Session"},
            "ep1",
            {"consistency_level": ConsistencyLevel.Session, "guarantee_timestamp": 2000},
            False,
            id="session-cached",
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
    ],
)
def test_construct_guarantee_ts(kwargs, endpoint, expected, use_default):
    ts_utils.update_collection_ts("coll1", 2000, "ep1", "db1")
    request_args = dict(kwargs)

    result = ts_utils.construct_guarantee_ts("coll1", request_args, endpoint, "db1")

    assert result is use_default
    assert request_args == expected
