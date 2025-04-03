from unittest.mock import patch

import pytest

from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        with patch("pymilvus.orm.utility.get_server_type", return_value="milvus"), patch('pymilvus.milvus_client.milvus_client.MilvusClient._create_connection', return_value="test"):
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
