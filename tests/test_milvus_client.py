import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import DataType
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            if isinstance(index_params, IndexParams):
                with pytest.raises(
                    ParamError, match="IndexParams is empty, no index can be created"
                ):
                    client.create_index("test_collection", index_params)
            elif index_params is None:
                with pytest.raises(ParamError, match=r"missing required argument:.*"):
                    client.create_index("test_collection", index_params)
            else:
                with pytest.raises(ParamError, match=r"wrong type of argument .*"):
                    client.create_index("test_collection", index_params)

    def test_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        assert len(index_params) == 0

        index_params.add_index("vector", index_type="FLAT", metric_type="L2")
        assert len(index_params) == 1

        index_params.add_index("vector2", index_type="HNSW", efConstruction=100, metric_type="L2")

        log.info(index_params)
        assert len(index_params) == 2

        for index in index_params:
            log.info(index)

    def test_connection_reuse(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch("pymilvus.orm.connections.Connections.connect", return_value=None), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ):
            client = MilvusClient()
            assert client._using == "http://localhost:19530"
            client = MilvusClient(user="test", password="foobar")
            assert client._using == "http://localhost:19530-test"
            client = MilvusClient(token="foobar")
            assert client._using == "http://localhost:19530-3858f62230ac3c915f300c664312c63f"

    @pytest.mark.parametrize(
        "data_type",
        [
            "FLOAT_VECTOR",
            "BINARY_VECTOR",
            "FLOAT16_VECTOR",
            "BFLOAT16_VECTOR",
            "SPARSE_FLOAT_VECTOR",
            "INT8_VECTOR",
        ],
    )
    def test_add_collection_field_vector_requires_nullable(self, data_type):
        """Test that adding vector field to collection requires nullable=True"""

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            dtype = getattr(DataType, data_type)

            # Should raise ParamError when nullable is not set or False
            with pytest.raises(
                ParamError,
                match="Adding vector field to existing collection requires nullable=True",
            ):
                client.add_collection_field(
                    collection_name="test_collection",
                    field_name="vector_field",
                    data_type=dtype,
                    dim=128,
                )

            # Should raise ParamError when nullable is explicitly False
            with pytest.raises(
                ParamError,
                match="Adding vector field to existing collection requires nullable=True",
            ):
                client.add_collection_field(
                    collection_name="test_collection",
                    field_name="vector_field",
                    data_type=dtype,
                    dim=128,
                    nullable=False,
                )

    def test_add_collection_field_vector_with_nullable_true(self):
        """Test that adding vector field with nullable=True passes validation"""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_conn = MagicMock()

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch.object(
            MilvusClient, "_get_connection", return_value=mock_conn
        ):
            client = MilvusClient()

            # Should not raise when nullable=True
            client.add_collection_field(
                collection_name="test_collection",
                field_name="vector_field",
                data_type=DataType.FLOAT_VECTOR,
                dim=128,
                nullable=True,
            )
            mock_conn.add_collection_field.assert_called_once()

    def test_add_collection_field_non_vector_no_nullable_required(self):
        """Test that non-vector fields don't require nullable=True"""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_conn = MagicMock()

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch.object(
            MilvusClient, "_get_connection", return_value=mock_conn
        ):
            client = MilvusClient()

            # Non-vector types should not require nullable
            client.add_collection_field(
                collection_name="test_collection",
                field_name="int_field",
                data_type=DataType.INT64,
            )
            mock_conn.add_collection_field.assert_called_once()

    def test_client_db_isolation(self):
        """
        Test that two clients sharing the same connection but using different databases
        remain isolated when one switches database.
        """
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client._utils.create_connection", return_value="shared_alias"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch(
            "pymilvus.orm.connections.Connections.has_connection", return_value=True
        ):
            client_a = MilvusClient(uri="http://localhost:19530", db_name="default")
            client_b = MilvusClient(uri="http://localhost:19530", db_name="testdb")

            assert client_a._db_name == "default"
            assert client_b._db_name == "testdb"

            # Mock describe_database to simulate that 'db1' exists
            # use_database now validates database existence by calling describe_database
            with patch.object(client_a, "describe_database", return_value={}):
                client_a.use_database("db1")

            assert client_a._db_name == "db1"
            assert client_b._db_name == "testdb"

            client_b.list_collections()

            assert mock_handler.list_collections.called
            _, kwargs = mock_handler.list_collections.call_args
            context = kwargs.get("context")

            assert context is not None
            assert context.get_db_name() == "testdb"
