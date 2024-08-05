import os
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pytest

from pymilvus.milvus_client import MilvusClient
from pymilvus.exceptions import ConnectionConfigException


@pytest.mark.skipif(sys.platform.startswith('win'), reason="Milvus Lite is not supported on Windows")
class TestMilvusLite:
    def test_milvus_lite(self):
        with TemporaryDirectory(dir='./') as root:
            db_file = os.path.join(root, 'test.db')
            client = MilvusClient(db_file)
            client.create_collection(
                collection_name="demo_collection",
                dimension=3
            )

            # Text strings to search from.
            docs = [
                "Artificial intelligence was founded as an academic discipline in 1956.",
                "Alan Turing was the first person to conduct substantial research in AI.",
                "Born in Maida Vale, London, Turing was raised in southern England.",
            ]

            vectors = [[np.random.uniform(-1, 1) for _ in range(3) ] for _ in range(len(docs))]
            data = [{"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors))]
            res = client.insert(
                collection_name="demo_collection",
                data=data
            )
            assert res["insert_count"] == 3

            res = client.search(
                    collection_name="demo_collection",
                    data=[vectors[0]],
                    filter="subject == 'history'",
                    limit=2,
                    output_fields=["text", "subject"],
                )
            assert len(res[0]) == 2

            # a query that retrieves all entities matching filter expressions.
            res = client.query(
                    collection_name="demo_collection",
                    filter="subject == 'history'",
                    output_fields=["text", "subject"],
                )
            assert len(res) == 3

            # delete
            res = client.delete(
                    collection_name="demo_collection",
                    filter="subject == 'history'",
                )
            assert len(res) == 3

    def test_illegal_name(self):
        try:
            MilvusClient("localhost")
            assert False
        except ConnectionConfigException as e:
            assert e.message == "uri: localhost is illegal, needs start with [unix, http, https, tcp] or a local file endswith [.db]"
