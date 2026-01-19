"""
MinHash Text Deduplication Demo

This example shows an end-to-end workflow to deduplicate texts using
MinHash Function + MINHASH_LSH in Milvus.

Requirements:
- Milvus server with MinHash function support (v2.5+)
- pymilvus with MINHASH FunctionType support

Constraints:
- Index Type: MINHASH_LSH
- Metric Type: MHJACCARD
- Output Field: BINARY_VECTOR with dim = num_hashes * 32
"""

from pymilvus import DataType, MilvusClient
from pymilvus.client.types import FunctionType
from pymilvus.orm.schema import Function


def create_dedup_collection(client: MilvusClient, collection_name: str):
    """Create a collection with MinHash function for deduplication."""
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=512)

    schema.add_function(
        Function(
            name="text_to_minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_signature"],
            params={
                "num_hashes": 16,
                "shingle_size": 3,
                "token_level": "word",
            },
        )
    )

    client.create_collection(collection_name=collection_name, schema=schema)


def create_dedup_index(client: MilvusClient, collection_name: str):
    """Create a MINHASH_LSH index with raw data for exact Jaccard reranking."""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="minhash_signature",
        index_type="MINHASH_LSH",
        metric_type="MHJACCARD",
        params={
            "mh_lsh_band": 8,
            "with_raw_data": True,
        },
    )
    client.create_index(collection_name, index_params)


def deduplicate_texts(
    client: MilvusClient,
    collection_name: str,
    texts: list,
    similarity_threshold: float = 0.8,
    top_k: int = 5,
):
    """Return unique texts and duplicate groups using MinHash + Jaccard similarity.

    similarity_threshold is MHJACCARD similarity (higher = more similar).
    """
    documents = [{"id": idx + 1, "text": text} for idx, text in enumerate(texts)]
    client.insert(collection_name, documents)
    client.flush(collection_name)
    client.load_collection(collection_name)

    duplicates = []
    unique_ids = set()

    for doc in documents:
        results = client.search(
            collection_name=collection_name,
            data=[doc["text"]],
            anns_field="minhash_signature",
            search_params={
                "metric_type": "MHJACCARD",
                "params": {
                    "mh_search_with_jaccard": True,
                    "refine_k": 50,
                },
            },
            limit=top_k,
            output_fields=["id", "text"],
        )

        is_duplicate = False
        for hit in results[0]:
            if hit["id"] == doc["id"]:
                continue
            if hit["distance"] >= similarity_threshold and hit["id"] < doc["id"]:
                duplicates.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "duplicate_of": hit["id"],
                        "similarity": hit["distance"],
                    }
                )
                is_duplicate = True
                break

        if not is_duplicate:
            unique_ids.add(doc["id"])

    unique_texts = [doc["text"] for doc in documents if doc["id"] in unique_ids]
    return unique_texts, duplicates


def main():
    uri = "http://localhost:19530"
    client = MilvusClient(uri=uri)

    collection_name = "minhash_dedup_demo"
    create_dedup_collection(client, collection_name)
    create_dedup_index(client, collection_name)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox jumped over a lazy dog.",
        "The fast brown fox leaps over the sleepy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Deep learning transforms artificial intelligence research.",
        "Completely unrelated text about cooking recipes.",
        "Completely unrelated text about cooking recipes!",
    ]

    unique_texts, duplicates = deduplicate_texts(
        client,
        collection_name,
        texts,
        similarity_threshold=0.8,
        top_k=5,
    )

    print("\nUnique texts:")
    for text in unique_texts:
        print(f"  - {text}")

    print("\nDuplicates:")
    if not duplicates:
        print("  (none)")
    else:
        for dup in duplicates:
            print(
                f"  - ID {dup['id']} is duplicate of ID {dup['duplicate_of']}, "
                f"similarity={dup['similarity']:.4f}"
            )

    client.drop_collection(collection_name)


if __name__ == "__main__":
    main()
