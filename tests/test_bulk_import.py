import sys
from unittest.mock import MagicMock, patch

import pymilvus.bulk_writer.bulk_import  # noqa: F401
import pytest
from pymilvus.bulk_writer.bulk_import import (
    _http_headers,
    _post_request,
    bulk_import,
    get_import_progress,
    list_import_jobs,
)
from pymilvus.exceptions import MilvusException

bulk_import_mod = sys.modules["pymilvus.bulk_writer.bulk_import"]


class TestHttpHeaders:
    def test_without_db_name(self):
        headers = _http_headers(api_key="my-key")
        assert headers["Authorization"] == "Bearer my-key"
        assert "DB-Name" not in headers

    def test_with_empty_db_name(self):
        headers = _http_headers(api_key="my-key", db_name="")
        assert "DB-Name" not in headers

    def test_with_db_name(self):
        headers = _http_headers(api_key="my-key", db_name="my_db")
        assert headers["DB-Name"] == "my_db"
        assert headers["Authorization"] == "Bearer my-key"


class TestPostRequest:
    @patch.object(bulk_import_mod.requests, "post")
    def test_pops_db_name_and_adds_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        resp = _post_request(
            url="http://example.com/api",
            api_key="my-key",
            params={"foo": "bar"},
            db_name="my_db",
        )

        assert resp is mock_resp
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["DB-Name"] == "my_db"
        assert "db_name" not in kwargs

    @patch.object(bulk_import_mod.requests, "post")
    def test_without_db_name_has_no_db_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        _post_request(
            url="http://example.com/api",
            api_key="my-key",
            params={"foo": "bar"},
        )

        _, kwargs = mock_post.call_args
        assert "DB-Name" not in kwargs["headers"]


class TestGetImportProgress:
    @patch.object(bulk_import_mod.requests, "post")
    def test_sends_db_name_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        resp = get_import_progress(
            url="http://example.com",
            job_id="job-123",
            api_key="my-key",
            db_name="my_db",
        )

        assert resp is mock_resp
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["url"] == "http://example.com/v2/vectordb/jobs/import/describe"
        assert kwargs["headers"]["DB-Name"] == "my_db"
        assert kwargs["json"] == {"jobId": "job-123", "clusterId": ""}
        assert "db_name" not in kwargs

    @patch.object(bulk_import_mod.requests, "post")
    def test_without_db_name_omits_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        get_import_progress(
            url="http://example.com",
            job_id="job-123",
            api_key="my-key",
        )

        _, kwargs = mock_post.call_args
        assert "DB-Name" not in kwargs["headers"]

    @patch.object(bulk_import_mod.requests, "post")
    def test_non_zero_code_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 1, "message": "boom"}
        mock_post.return_value = mock_resp

        with pytest.raises(MilvusException, match="boom"):
            get_import_progress(
                url="http://example.com",
                job_id="job-123",
                api_key="my-key",
                db_name="my_db",
            )


class TestBulkImport:
    @patch.object(bulk_import_mod.requests, "post")
    def test_sends_db_name_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        resp = bulk_import(
            url="http://example.com",
            collection_name="my_collection",
            api_key="my-key",
            db_name="my_db",
            files=[["file1.parquet"]],
        )

        assert resp is mock_resp
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["url"] == "http://example.com/v2/vectordb/jobs/import/create"
        assert kwargs["headers"]["DB-Name"] == "my_db"
        assert kwargs["json"]["dbName"] == "my_db"
        assert kwargs["json"]["collectionName"] == "my_collection"
        assert "db_name" not in kwargs

    @patch.object(bulk_import_mod.requests, "post")
    def test_without_db_name_has_no_db_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        bulk_import(
            url="http://example.com",
            collection_name="my_collection",
            api_key="my-key",
            files=[["file1.parquet"]],
        )

        _, kwargs = mock_post.call_args
        assert "DB-Name" not in kwargs["headers"]
        assert kwargs["json"]["dbName"] == ""


class TestListImportJobs:
    @patch.object(bulk_import_mod.requests, "post")
    def test_sends_db_name_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        resp = list_import_jobs(
            url="http://example.com",
            collection_name="my_collection",
            api_key="my-key",
            db_name="my_db",
        )

        assert resp is mock_resp
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["url"] == "http://example.com/v2/vectordb/jobs/import/list"
        assert kwargs["headers"]["DB-Name"] == "my_db"
        assert kwargs["json"]["dbName"] == "my_db"
        assert kwargs["json"]["collectionName"] == "my_collection"
        assert "db_name" not in kwargs

    @patch.object(bulk_import_mod.requests, "post")
    def test_without_db_name_has_no_db_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "data": {}}
        mock_post.return_value = mock_resp

        list_import_jobs(
            url="http://example.com",
            collection_name="my_collection",
            api_key="my-key",
        )

        _, kwargs = mock_post.call_args
        assert "DB-Name" not in kwargs["headers"]
        assert kwargs["json"]["dbName"] == ""
