"""Shared fixtures and test data for GrpcHandler tests."""

from unittest.mock import MagicMock

import pytest
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2

# ============================================================
# Test Data Tables
# ============================================================

COLLECTION_VALIDATION_CASES = [
    pytest.param("", ParamError, id="empty_name"),
    pytest.param(None, ParamError, id="none_name"),
]

PARTITION_VALIDATION_CASES = [
    pytest.param("test_coll", "test_part", None, id="valid"),
    pytest.param("", "test_part", ParamError, id="empty_collection"),
]

HAS_COLLECTION_RESPONSE_CASES = [
    pytest.param(0, 0, "", True, id="exists"),
    pytest.param(common_pb2.CollectionNotExists, 0, "", False, id="not_found_error_code"),
    pytest.param(0, 100, "", False, id="not_found_status_code"),
    pytest.param(
        common_pb2.UnexpectedError, 0, "can't find collection", False, id="unexpected_reason"
    ),
]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def handler():
    """Create a GrpcHandler with mocked channel and stub."""
    h = GrpcHandler(channel=MagicMock())
    h._stub = MagicMock()
    return h


@pytest.fixture
def mock_schema():
    """Standard mock schema for tests."""
    return {
        "fields": [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
        ],
        "enable_dynamic_field": False,
    }


# ============================================================
# Helper Functions
# ============================================================


def make_status(code=0, error_code=0, reason=""):
    """Create a mock status response."""
    status = MagicMock()
    status.code = code
    status.error_code = error_code
    status.reason = reason
    return status


def make_response(code=0, error_code=0, reason="", **kwargs):
    """Create a mock response with status and additional fields."""
    resp = MagicMock()
    resp.status.code = code
    resp.status.error_code = error_code
    resp.status.reason = reason
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


def make_mutation_response(insert_cnt=0, ids=None, upsert_cnt=0, delete_cnt=0):
    """Create a mock mutation response."""
    mock_resp = MagicMock()
    mock_resp.status.code = 0
    mock_resp.status.error_code = 0
    mock_resp.status.reason = ""
    mock_resp.insert_cnt = insert_cnt
    mock_resp.delete_cnt = delete_cnt
    mock_resp.upsert_cnt = upsert_cnt
    mock_resp.timestamp = 123456
    mock_resp.succ_index = list(range(len(ids or [])))
    mock_resp.err_index = []
    mock_resp.IDs.WhichOneof.return_value = "int_id"
    mock_resp.IDs.int_id.data = ids or []
    return mock_resp
