"""search_iterator over an emb_list (array-of-vector) field, and a hybrid
search_iterator that fuses an emb_list modality with a dense modality via RRF.

An emb_list field stores, per row, an array of vectors (ColBERT-style late
chunking: one row per document, many paragraph vectors per row). Its similarity
metric is MAX_SIM -- a sum-of-max over paragraph similarities -- so a query is
itself a *list* of vectors, passed as one EmbeddingList.

Run against a local milvus (http://localhost:19530).
"""

import numpy as np

from pymilvus import DataType, MilvusClient
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.embedding_list import EmbeddingList

COLLECTION_NAME = "emblist_iterator_demo"
DENSE_DIM = 16
CHUNK_DIM = 32
NUM_ROWS = 5000
rng = np.random.default_rng(seed=19530)


def build_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DENSE_DIM)

    # the emb_list field: an array of structs, each holding one chunk vector
    chunk_schema = client.create_struct_field_schema()
    chunk_schema.add_field("chunk_vec", DataType.FLOAT_VECTOR, dim=CHUNK_DIM)
    schema.add_field(
        "chunks",
        datatype=DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=chunk_schema,
        max_capacity=20,
    )
    client.create_collection(COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE")
    index_params.add_index(
        field_name="chunks[chunk_vec]",
        index_type="HNSW",
        index_name="chunk_vec_index",
        metric_type="MAX_SIM_COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )
    client.create_index(COLLECTION_NAME, schema=schema, index_params=index_params)

    rows = []
    for i in range(NUM_ROWS):
        n_chunks = int(rng.integers(2, 8))
        rows.append(
            {
                "title": f"doc-{i}",
                "embedding": rng.random(DENSE_DIM, dtype=np.float32),
                "chunks": [
                    {"chunk_vec": rng.random(CHUNK_DIM, dtype=np.float32)}
                    for _ in range(n_chunks)
                ],
            }
        )
    client.insert(COLLECTION_NAME, rows)
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    print(f"built {COLLECTION_NAME}: {NUM_ROWS} rows")


def emblist_search_iterator(client: MilvusClient):
    """Iterate an emb_list field. One query = several vectors in one EmbeddingList."""
    query = EmbeddingList([rng.random(CHUNK_DIM, dtype=np.float32) for _ in range(4)])

    iterator = client.search_iterator(
        collection_name=COLLECTION_NAME,
        data=[query],  # a single emb_list query
        anns_field="chunks[chunk_vec]",
        batch_size=200,
        output_fields=["title"],
    )
    total, page_idx = 0, 0
    while True:
        page = iterator.next()
        if len(page) == 0:
            iterator.close()
            break
        total += len(page)
        page_idx += 1
        print(f"  emb_list page {page_idx}: {len(page)} hits (top: {page[0]['title']})")
    print(f"emb_list search_iterator: {total} hits over {page_idx} pages")


def hybrid_search_iterator(client: MilvusClient):
    """Fuse a dense modality and the emb_list modality with RRF, streamed."""
    dense_req = AnnSearchRequest(
        data=[rng.random(DENSE_DIM, dtype=np.float32)],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=200,
    )
    emblist_req = AnnSearchRequest(
        data=[EmbeddingList([rng.random(CHUNK_DIM, dtype=np.float32) for _ in range(4)])],
        anns_field="chunks[chunk_vec]",
        param={"metric_type": "MAX_SIM_COSINE"},
        limit=200,
    )
    iterator = client.hybrid_search_iterator(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, emblist_req],
        batch_size=200,
        rrf_k=60,
        output_fields=["title"],
    )
    total, page_idx = 0, 0
    while True:
        page = iterator.next()
        if len(page) == 0:
            iterator.close()
            break
        total += len(page)
        page_idx += 1
        # the emitted score is the NRA lower bound on the true RRF score
        print(f"  hybrid page {page_idx}: {len(page)} hits (top: {page[0]['title']})")
    print(f"hybrid search_iterator: {total} fused hits over {page_idx} pages")


def main():
    client = MilvusClient("http://localhost:19530")
    build_collection(client)
    emblist_search_iterator(client)
    hybrid_search_iterator(client)


if __name__ == "__main__":
    main()
