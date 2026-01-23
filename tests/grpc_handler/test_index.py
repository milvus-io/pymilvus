"""Tests for GrpcHandler index and alias operations."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import MilvusException, ParamError

from .conftest import make_response, make_status


class TestGrpcHandlerIndexOps:
    """Tests for index operations."""

    def test_create_index(self, handler):
        handler._stub.CreateIndex.future.return_value.result.return_value = make_status()
        with patch.object(handler, "wait_for_creating_index", return_value=(True, "")):
            with patch.object(handler, "alloc_timestamp", return_value=123):
                handler.create_index("coll", "vec", {"index_type": "IVF_FLAT"})
                handler._stub.CreateIndex.future.assert_called_once()

    def test_create_index_async(self, handler):
        mock_future = MagicMock()
        handler._stub.CreateIndex.future.return_value = mock_future

        with patch.object(handler, "alloc_timestamp", return_value=123):
            result = handler.create_index("coll", "vec", {"index_type": "IVF_FLAT"}, _async=True)
            assert result is not None

    def test_create_index_sync_fails(self, handler):
        handler._stub.CreateIndex.future.return_value.result.return_value = make_status()

        with patch.object(handler, "alloc_timestamp", return_value=123):
            with patch.object(
                handler, "wait_for_creating_index", return_value=(False, "index failed")
            ):
                with pytest.raises(MilvusException):
                    handler.create_index("coll", "vec", {"index_type": "IVF_FLAT"})

    def test_drop_index(self, handler):
        handler._stub.DropIndex.return_value = make_status()
        handler.drop_index("coll", "vec", "idx")
        handler._stub.DropIndex.assert_called_once()

    def test_describe_index(self, handler):
        mock_idx = MagicMock(field_name="vec", index_name="idx", params=[])
        handler._stub.DescribeIndex.return_value = make_response(index_descriptions=[mock_idx])
        result = handler.describe_index("coll", "idx")
        assert result is not None

    def test_list_indexes(self, handler):
        mock_idx = MagicMock(field_name="vec", index_name="idx")
        handler._stub.DescribeIndex.return_value = make_response(index_descriptions=[mock_idx])
        result = handler.list_indexes("coll")
        assert len(result) == 1

    def test_get_index_state(self, handler):
        mock_idx = MagicMock()
        mock_idx.state = 3  # Finished
        mock_idx.index_state_fail_reason = ""
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx]
        handler._stub.DescribeIndex.return_value = mock_resp

        state, _reason = handler.get_index_state("coll", "idx")
        assert state == 3

    def test_get_index_state_with_field_name(self, handler):
        mock_idx1 = MagicMock()
        mock_idx1.field_name = "vec1"
        mock_idx1.state = 1
        mock_idx1.index_state_fail_reason = ""
        mock_idx2 = MagicMock()
        mock_idx2.field_name = "vec2"
        mock_idx2.state = 3
        mock_idx2.index_state_fail_reason = ""
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx1, mock_idx2]
        handler._stub.DescribeIndex.return_value = mock_resp

        state, _reason = handler.get_index_state("coll", "idx", field_name="vec2")
        assert state == 3

    def test_get_index_build_progress(self, handler):
        mock_idx = MagicMock()
        mock_idx.total_rows = 1000
        mock_idx.indexed_rows = 500
        mock_idx.pending_index_rows = 500
        mock_idx.state = 1  # InProgress
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx]
        handler._stub.DescribeIndex.return_value = mock_resp
        result = handler.get_index_build_progress("coll", "idx")
        assert result is not None
        assert result["total_rows"] == 1000

    def test_wait_for_creating_index_complete(self, handler):
        mock_idx = MagicMock()
        mock_idx.state = 3  # Finished
        mock_idx.index_state_fail_reason = ""
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx]
        handler._stub.DescribeIndex.return_value = mock_resp
        handler._stub.AllocTimestamp.return_value = make_response(timestamp=123)
        with patch("time.sleep"):
            result, _reason = handler.wait_for_creating_index("coll", "idx")
            assert result is True

    def test_wait_for_creating_index_failed(self, handler):
        mock_idx = MagicMock()
        mock_idx.state = 4  # Failed
        mock_idx.index_state_fail_reason = "Build failed"
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx]
        handler._stub.DescribeIndex.return_value = mock_resp
        handler._stub.AllocTimestamp.return_value = make_response(timestamp=123)

        with patch("time.sleep"):
            result, reason = handler.wait_for_creating_index("coll", "idx")
            assert result is False
            assert reason == "Build failed"


class TestGrpcHandlerIndexProperties:
    """Tests for index property operations."""

    def test_alter_index_properties(self, handler):
        handler._stub.AlterIndex.return_value = make_status()
        handler.alter_index_properties("coll", "idx", {"mmap.enabled": "true"})
        handler._stub.AlterIndex.assert_called_once()

    def test_alter_index_properties_none_raises(self, handler):
        with pytest.raises(ParamError):
            handler.alter_index_properties("coll", "idx", None)

    def test_drop_index_properties(self, handler):
        handler._stub.AlterIndex.return_value = make_status()
        handler.drop_index_properties("coll", "idx", ["mmap.enabled"])
        handler._stub.AlterIndex.assert_called_once()


class TestGrpcHandlerAliasOps:
    """Tests for alias operations."""

    def test_create_alias(self, handler):
        handler._stub.CreateAlias.return_value = make_status()
        handler.create_alias("coll", "alias")
        handler._stub.CreateAlias.assert_called_once()

    def test_drop_alias(self, handler):
        handler._stub.DropAlias.return_value = make_status()
        handler.drop_alias("alias")
        handler._stub.DropAlias.assert_called_once()

    def test_alter_alias(self, handler):
        handler._stub.AlterAlias.return_value = make_status()
        handler.alter_alias("coll", "alias")
        handler._stub.AlterAlias.assert_called_once()

    def test_list_aliases(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.aliases = ["a1", "a2"]
        mock_resp.collection_name = "coll"
        mock_resp.db_name = "default"
        handler._stub.ListAliases.return_value = mock_resp
        result = handler.list_aliases("coll")
        assert result["aliases"] == ["a1", "a2"]

    def test_describe_alias(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.alias = "test_alias"
        mock_resp.collection = "test_coll"
        mock_resp.db_name = "default"
        handler._stub.DescribeAlias.return_value = mock_resp
        result = handler.describe_alias("test_alias")
        assert result is not None
