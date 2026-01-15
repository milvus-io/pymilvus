import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        
        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            if isinstance(index_params, IndexParams):
                with pytest.raises(ParamError, match="IndexParams is empty, no index can be created"):
                    client.create_index("test_collection", index_params)
            elif index_params is None:
                with pytest.raises(ParamError, match="missing required argument:.*"):
                    client.create_index("test_collection", index_params)
            else:
                with pytest.raises(ParamError, match="wrong type of argument .*"):
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
        
        with patch("pymilvus.orm.connections.Connections.connect", return_value=None), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
        from pymilvus.orm.types import DataType

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ):
            client = MilvusClient()
            dtype = getattr(DataType, data_type)

            # Should raise ParamError when nullable is not set or False
            with pytest.raises(
                ParamError, match="Adding vector field to existing collection requires nullable=True"
            ):
                client.add_collection_field(
                    collection_name="test_collection",
                    field_name="vector_field",
                    data_type=dtype,
                    dim=128,
                )

            # Should raise ParamError when nullable is explicitly False
            with pytest.raises(
                ParamError, match="Adding vector field to existing collection requires nullable=True"
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
        from pymilvus.orm.types import DataType

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
        from pymilvus.orm.types import DataType

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
