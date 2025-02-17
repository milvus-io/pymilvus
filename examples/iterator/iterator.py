from pymilvus.client.abstract import Hits
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
)
import numpy as np

collection_name = "test_milvus_client_iterator"
prepare_new_data = True
clean_exist = True

USER_ID = "id"
AGE = "age"
DEPOSIT = "deposit"
PICTURE = "picture"
DIM = 8
NUM_ENTITIES = 10000
rng = np.random.default_rng(seed=19530)


def test_query_iterator(milvus_client: MilvusClient):
    # test query iterator
    expr = f"10 <= {AGE} <= 25"
    output_fields = [USER_ID, AGE]
    queryIt = milvus_client.query_iterator(collection_name, filter=expr, batch_size=50, output_fields=output_fields)
    page_idx = 0
    while True:
        res = queryIt.next()
        if len(res) == 0:
            print("query iteration finished, close")
            queryIt.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")

def test_search_iterator(milvus_client: MilvusClient):
    vector_to_search = rng.random((1, DIM), np.float32)
    search_iterator = milvus_client.search_iterator(collection_name, data=vector_to_search, batch_size=100, anns_field=PICTURE)

    page_idx = 0
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            print("search iteration finished, close")
            search_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")

def test_search_iterator_with_filter(milvus_client: MilvusClient):
    vector_to_search = rng.random((1, DIM), np.float32)
    expr = f"10 <= {AGE} <= 25"
    valid_ids = [1, 12, 123, 1234]

    def external_filter_func(hits: Hits):
        # option 1
        return list(filter(lambda hit: hit.id in valid_ids, hits))

        # option 2
        results = []
        for hit in hits:
            if hit.id in valid_ids:
                results.append(hit)
        return results

    search_iterator = milvus_client.search_iterator(
        collection_name=collection_name,
        data=vector_to_search,
        batch_size=100,
        anns_field=PICTURE,
        filter=expr,
        external_filter_func=external_filter_func,
        output_fields=[USER_ID, AGE]
    )

    page_idx = 0
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            print("search iteration with external filter finished, close")
            search_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")

def main():
    milvus_client = MilvusClient("http://localhost:19530")
    if milvus_client.has_collection(collection_name) and clean_exist:
        milvus_client.drop_collection(collection_name)
        print(f"dropped existed collection{collection_name}")

    if not milvus_client.has_collection(collection_name):
        fields = [
            FieldSchema(name=USER_ID, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=AGE, dtype=DataType.INT64),
            FieldSchema(name=DEPOSIT, dtype=DataType.DOUBLE),
            FieldSchema(name=PICTURE, dtype=DataType.FLOAT_VECTOR, dim=DIM)
        ]
        schema = CollectionSchema(fields)
        milvus_client.create_collection(collection_name, dimension=DIM, schema=schema)

    if prepare_new_data:
        entities = []
        for i in range(NUM_ENTITIES):
            entity = {
                USER_ID: i,
                AGE: (i % 100),
                DEPOSIT: float(i),
                PICTURE: rng.random((1, DIM))[0]
            }
            entities.append(entity)
        milvus_client.insert(collection_name, entities)
        milvus_client.flush(collection_name)
        print(f"Finish flush collections:{collection_name}")

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name=PICTURE,
        index_type='IVF_FLAT',
        metric_type='L2',
        params={"nlist": 1024}
    )
    milvus_client.create_index(collection_name, index_params)
    milvus_client.load_collection(collection_name)
    test_query_iterator(milvus_client=milvus_client)
    test_search_iterator(milvus_client=milvus_client)
    test_search_iterator_with_filter(milvus_client=milvus_client)


if __name__ == '__main__':
    main()
