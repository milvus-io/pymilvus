import sys

import numpy as np
import pytest
from pymilvus.exceptions import ConnectionConfigException
from pymilvus.milvus_client import MilvusClient

milvus_lite = pytest.importorskip("milvus_lite", reason="milvus-lite not installed")


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Milvus Lite is not supported on Windows"
)
class TestMilvusLite:
    def test_milvus_client_with_local_db_path(self, tmp_path):
        """MilvusClient("./test.db") should connect via Milvus Lite.

        Regression test for https://github.com/milvus-io/pymilvus/issues/3314
        and https://github.com/milvus-io/pymilvus/issues/3317.

        On pymilvus 2.6.10 this hangs with 'dns:///' gRPC error because the
        ConnectionManager constructs an empty gRPC target URI for local .db paths.
        """
        db_file = tmp_path / "test.db"
        client = MilvusClient(db_file.as_posix(), timeout=10)
        try:
            collections = client.list_collections()
            assert isinstance(collections, list)
        finally:
            client.close()

    def test_milvus_lite_insert_search(self, tmp_path):
        """End-to-end test: create collection, insert, search, query, delete via Milvus Lite."""
        db_file = tmp_path / "test.db"
        client = MilvusClient(db_file.as_posix(), timeout=10)
        try:
            client.create_collection(collection_name="demo_collection", dimension=3)

            rng = np.random.default_rng(seed=19530)
            vectors = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
            data = [
                {"id": i, "vector": vectors[i], "text": f"doc_{i}", "subject": "history"}
                for i in range(len(vectors))
            ]
            res = client.insert(collection_name="demo_collection", data=data)
            assert res["insert_count"] == 3

            res = client.search(
                collection_name="demo_collection",
                data=[vectors[0]],
                filter="subject == 'history'",
                limit=2,
                output_fields=["text", "subject"],
            )
            assert len(res[0]) == 2

            res = client.query(
                collection_name="demo_collection",
                filter="subject == 'history'",
                output_fields=["text", "subject"],
            )
            assert len(res) == 3

            res = client.delete(
                collection_name="demo_collection",
                filter="subject == 'history'",
            )
            assert len(res) == 3
        finally:
            client.close()

    def test_milvus_lite_multiple_clients_same_db(self, tmp_path):
        """Two MilvusClient instances sharing the same .db file should work."""
        db_file = tmp_path / "shared.db"
        client1 = MilvusClient(db_file.as_posix(), timeout=10)
        try:
            client1.create_collection(collection_name="col1", dimension=3)
            assert "col1" in client1.list_collections()

            client2 = MilvusClient(db_file.as_posix(), timeout=10)
            try:
                assert "col1" in client2.list_collections()
            finally:
                client2.close()
        finally:
            client1.close()

    def test_illegal_name(self):
        with pytest.raises(ConnectionConfigException) as e:
            MilvusClient("localhost")
        assert (
            e.value.message
            == "uri: localhost is illegal, needs start with [unix, http, https, tcp] or a local file endswith [.db]"
        )
