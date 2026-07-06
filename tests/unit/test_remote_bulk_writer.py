from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from pymilvus.bulk_writer.constants import BulkFileType
from pymilvus.bulk_writer.remote_bulk_writer import RemoteBulkWriter
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema


class TestRemoteBulkWriter:
    def _simple_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]
        return CollectionSchema(fields=fields)

    @patch("pymilvus.bulk_writer.remote_bulk_writer.Minio")
    def test_enable_virtual_style_endpoint_when_flag_true(self, mock_minio, tmp_path):
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        connect_param = RemoteBulkWriter.S3ConnectParam(
            bucket_name="test-bucket",
            endpoint="localhost:9000",
            access_key="ak",
            secret_key="sk",
            secure=True,
            enable_virtual_style_endpoint=True,
            session_token="token",
            region="us-west-2",
        )

        RemoteBulkWriter(
            schema=self._simple_schema(),
            remote_path="bulk/test",
            connect_param=connect_param,
            local_path=str(tmp_path),
        )

        mock_minio.assert_called_once()
        mock_client.enable_virtual_style_endpoint.assert_called_once()

    @patch("pymilvus.bulk_writer.remote_bulk_writer.Minio")
    def test_not_enable_virtual_style_endpoint_when_flag_false(self, mock_minio, tmp_path):
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        connect_param = RemoteBulkWriter.S3ConnectParam(
            bucket_name="test-bucket",
            endpoint="localhost:9000",
            access_key="ak",
            secret_key="sk",
            enable_virtual_style_endpoint=False,
        )

        RemoteBulkWriter(
            schema=self._simple_schema(),
            remote_path="bulk/test",
            connect_param=connect_param,
            local_path=str(tmp_path),
        )

        mock_minio.assert_called_once()
        mock_client.enable_virtual_style_endpoint.assert_not_called()

    @patch("pymilvus.bulk_writer.remote_bulk_writer.Minio")
    def test_minio_init_args_are_parsed_correctly(self, mock_minio, tmp_path):
        mock_minio.return_value = MagicMock()

        connect_param = RemoteBulkWriter.S3ConnectParam(
            bucket_name="test-bucket",
            endpoint="localhost:9000",
            access_key="ak",
            secret_key="sk",
            secure=True,
            session_token="token",
            region="us-west-2",
        )

        RemoteBulkWriter(
            schema=self._simple_schema(),
            remote_path="bulk/test",
            connect_param=connect_param,
            local_path=str(tmp_path),
        )

        _, kwargs = mock_minio.call_args
        assert kwargs["endpoint"] == "localhost:9000"
        assert kwargs["access_key"] == "ak"
        assert kwargs["secret_key"] == "sk"
        assert kwargs["secure"] is True
        assert kwargs["session_token"] == "token"
        assert kwargs["region"] == "us-west-2"

    def test_upload_jsonl_file(self, tmp_path):
        writer = RemoteBulkWriter(
            schema=self._simple_schema(),
            remote_path="bulk/test",
            connect_param=None,
            file_type=BulkFileType.JSONL,
            local_path=str(tmp_path),
        )
        jsonl_file = writer._local_path / "1.jsonl"
        jsonl_file.write_text('{"id":1,"vector":[1.0,2.0,3.0,4.0]}\n')

        with ExitStack() as stack:
            stack.enter_context(patch.object(writer, "_bucket_exists", return_value=True))
            stack.enter_context(patch.object(writer, "_object_exists", return_value=False))
            mock_upload_object = stack.enter_context(patch.object(writer, "_upload_object"))
            mock_local_rm = stack.enter_context(patch.object(writer, "_local_rm"))
            uploaded_files = writer._upload([str(jsonl_file)])

        assert len(uploaded_files) == 1
        assert uploaded_files[0].endswith("/1.jsonl")
        mock_upload_object.assert_called_once()
        _, kwargs = mock_upload_object.call_args
        assert kwargs["file_path"].endswith(".jsonl")
        assert kwargs["object_name"] == uploaded_files[0]
        assert kwargs["object_name"].endswith("/1.jsonl")
        assert "bulk/test/" in kwargs["object_name"]
        assert "\\" not in kwargs["object_name"]
        mock_local_rm.assert_called_once_with(str(jsonl_file))
        assert writer.batch_files[0][0] == uploaded_files[0]

    def test_upload_jsonl_file_with_relative_local_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        relative_local_path = Path("relative-bulk")

        writer = RemoteBulkWriter(
            schema=self._simple_schema(),
            remote_path="bulk/test",
            connect_param=None,
            file_type=BulkFileType.JSONL,
            local_path=str(relative_local_path),
        )
        jsonl_file = writer._local_path / "1.jsonl"
        jsonl_file.write_text('{"id":1,"vector":[1.0,2.0,3.0,4.0]}\n')

        with ExitStack() as stack:
            stack.enter_context(patch.object(writer, "_bucket_exists", return_value=True))
            stack.enter_context(patch.object(writer, "_object_exists", return_value=False))
            mock_upload_object = stack.enter_context(patch.object(writer, "_upload_object"))
            stack.enter_context(patch.object(writer, "_local_rm"))
            uploaded_files = writer._upload([str(jsonl_file)])

        assert len(uploaded_files) == 1
        _, kwargs = mock_upload_object.call_args
        assert kwargs["object_name"] == uploaded_files[0]
        assert kwargs["object_name"].endswith("/1.jsonl")
        assert "bulk/test/" in kwargs["object_name"]
        assert "\\" not in kwargs["object_name"]
