"""
MinHash Function Example - Text Deduplication and Similarity Search

This example demonstrates how to use MinHash Function in Milvus for:
1. Text deduplication (near-duplicate detection)
2. Approximate similarity search based on Jaccard distance

MinHash converts text into binary signatures that preserve Jaccard similarity,
enabling efficient nearest neighbor search for similar documents.

Requirements:
- Milvus server with MinHash function support (v2.5+)
- pymilvus with MINHASH FunctionType support
"""

from pymilvus import MilvusClient, DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema, Function
from pymilvus.client.types import FunctionType


def create_minhash_collection(client: MilvusClient, collection_name: str):
    """Create a collection with MinHash function for text deduplication.

    Schema:
    - id: Primary key (INT64)
    - text: Original text content (VARCHAR)
    - minhash_signature: MinHash binary vector (BINARY_VECTOR, dim=512)

    The MinHash function automatically converts text -> minhash_signature on insert.
    """
    # Check if collection exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    # Create schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # Add fields
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="minhash_signature",
        datatype=DataType.BINARY_VECTOR,
        dim=512,  # dim = num_hashes * 32 = 16 * 32 = 512
    )

    # Add MinHash function
    # The function automatically generates minhash signatures from text on insert
    minhash_function = Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            "num_hashes": 16,        # Number of hash functions (dim / 32)
            "shingle_size": 3,       # N-gram size for text shingling
            "hash_function": "xxhash64",  # Hash function: "xxhash64" or "sha1"
            "token_level": "word",   # Tokenization: "word" or "char"
        },
    )
    schema.add_function(minhash_function)

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    print(f"Created collection: {collection_name}")

    return schema


def create_minhash_index(client: MilvusClient, collection_name: str):
    """Create MinHash LSH index for efficient similarity search.

    IMPORTANT: MinHash Function output fields MUST use:
    - index_type: MINHASH_LSH
    - metric_type: MHJACCARD
    """
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="minhash_signature",
        index_type="MINHASH_LSH",     # Required for MinHash Function output
        metric_type="MHJACCARD",       # Required for MinHash Function output
        params={
            "mh_lsh_band": 8,           # LSH band count (affects recall/speed tradeoff)
            "with_raw_data": True,       # Keep raw data for exact reranking
        },
    )

    client.create_index(collection_name, index_params)
    print(f"Created MINHASH_LSH index on collection: {collection_name}")


def insert_documents(client: MilvusClient, collection_name: str):
    """Insert sample documents. MinHash signatures are auto-generated."""
    documents = [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
        {"id": 2, "text": "A quick brown fox jumped over a lazy dog."},  # Similar to 1
        {"id": 3, "text": "The fast brown fox leaps over the sleepy dog."},  # Similar to 1
        {"id": 4, "text": "Machine learning is transforming artificial intelligence."},
        {"id": 5, "text": "Deep learning transforms artificial intelligence research."},  # Similar to 4
        {"id": 6, "text": "Python is a popular programming language for data science."},
        {"id": 7, "text": "Data science uses Python as a popular language."},  # Similar to 6
        {"id": 8, "text": "The weather today is sunny and warm."},
        {"id": 9, "text": "Today's weather is warm and sunny."},  # Similar to 8
        {"id": 10, "text": "Completely unrelated text about cooking recipes."},
    ]

    # Note: minhash_signature field is NOT provided - it will be auto-generated
    client.insert(collection_name, documents)
    print(f"Inserted {len(documents)} documents")


def search_similar_documents(client: MilvusClient, collection_name: str, query_text: str, top_k: int = 5):
    """Search for similar documents using MinHash.

    The search uses MinHash signatures for approximate nearest neighbor search
    based on Jaccard similarity.
    """
    client.load_collection(collection_name)

    results = client.search(
        collection_name=collection_name,
        data=[query_text],  # Text query - will be converted to MinHash signature
        anns_field="minhash_signature",
        search_params={
            "metric_type": "MHJACCARD",
            "params": {
                "mh_search_with_jaccard": True,  # Use exact Jaccard for reranking
                "refine_k": 50,  # Candidate count for refinement
            },
        },
        limit=top_k,
        output_fields=["id", "text"],
    )

    return results


def find_duplicates(client: MilvusClient, collection_name: str, threshold: float = 0.3):
    """Find near-duplicate documents based on Jaccard distance threshold.

    Lower distance = more similar (0 = identical, 1 = completely different)
    Typical thresholds:
    - 0.1-0.2: Very similar (near-duplicates)
    - 0.3-0.4: Similar content
    - 0.5+: Different content
    """
    client.load_collection(collection_name)

    # Query all documents
    all_docs = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id", "text"],
        limit=1000,
    )

    duplicates = []

    for doc in all_docs:
        results = client.search(
            collection_name=collection_name,
            data=[doc["text"]],
            anns_field="minhash_signature",
            search_params={
                "metric_type": "MHJACCARD",
                "params": {"mh_search_with_jaccard": True},
            },
            limit=5,
            output_fields=["id", "text"],
        )

        for hit in results[0]:
            # Skip self-match
            if hit["id"] == doc["id"]:
                continue
            # Check distance threshold
            if hit["distance"] < threshold:
                pair = tuple(sorted([doc["id"], hit["id"]]))
                if pair not in [tuple(sorted([d["id1"], d["id2"]])) for d in duplicates]:
                    duplicates.append({
                        "id1": doc["id"],
                        "id2": hit["id"],
                        "text1": doc["text"],
                        "text2": hit["entity"]["text"],
                        "distance": hit["distance"],
                        "similarity": 1 - hit["distance"],  # Jaccard similarity
                    })

    return duplicates


def main():
    """Main example demonstrating MinHash function usage."""
    # Connect to Milvus
    uri = "http://localhost:19530"
    client = MilvusClient(uri=uri)

    collection_name = "minhash_demo"

    print("=" * 60)
    print("MinHash Function Example")
    print("=" * 60)

    # Step 1: Create collection with MinHash function
    print("\n[Step 1] Creating collection with MinHash function...")
    create_minhash_collection(client, collection_name)

    # Step 2: Create MinHash LSH index
    print("\n[Step 2] Creating MINHASH_LSH index...")
    create_minhash_index(client, collection_name)

    # Step 3: Insert documents
    print("\n[Step 3] Inserting documents...")
    insert_documents(client, collection_name)

    # Step 4: Search for similar documents
    print("\n[Step 4] Searching for similar documents...")
    query = "The fast brown fox jumps over the lazy dog."
    print(f"Query: '{query}'")
    print("-" * 40)

    results = search_similar_documents(client, collection_name, query, top_k=5)

    print("Top 5 similar documents:")
    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. ID={hit['id']}, Distance={hit['distance']:.4f}")
        print(f"     Text: {hit['entity']['text']}")

    # Step 5: Find duplicates
    print("\n[Step 5] Finding near-duplicate documents (threshold=0.3)...")
    print("-" * 40)

    duplicates = find_duplicates(client, collection_name, threshold=0.3)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate pairs:")
        for dup in duplicates:
            print(f"  - IDs: {dup['id1']} <-> {dup['id2']}")
            print(f"    Similarity: {dup['similarity']:.2%}")
            print(f"    Text1: {dup['text1'][:50]}...")
            print(f"    Text2: {dup['text2'][:50]}...")
            print()
    else:
        print("No duplicates found with threshold 0.3")

    # Cleanup
    print("\n[Cleanup] Dropping collection...")
    client.drop_collection(collection_name)
    print("Done!")


def example_schema_only():
    """Example showing schema creation without Milvus server.

    This example demonstrates how to define MinHash function in schema
    without connecting to a Milvus server.
    """
    print("=" * 60)
    print("MinHash Schema Definition Example (No Server Required)")
    print("=" * 60)

    # Method 1: Using CollectionSchema directly
    print("\n[Method 1] Using CollectionSchema and Function classes:")

    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("minhash_vector", DataType.BINARY_VECTOR, dim=512),
        ],
        functions=[
            Function(
                name="text_to_minhash",
                function_type=FunctionType.MINHASH,
                input_field_names=["text"],
                output_field_names=["minhash_vector"],
                params={
                    "num_hashes": 16,
                    "shingle_size": 3,
                    "hash_function": "xxhash64",
                    "token_level": "word",
                },
            )
        ],
    )

    print(f"Schema: {schema.to_dict()}")
    print(f"Functions: {[f.to_dict() for f in schema.functions]}")

    # Check output field is marked
    output_field = next(f for f in schema.fields if f.name == "minhash_vector")
    print(f"Output field is_function_output: {output_field.is_function_output}")

    # Method 2: Using add_function
    print("\n[Method 2] Using add_function method:")

    schema2 = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("content", DataType.VARCHAR, max_length=65535),
            FieldSchema("signature", DataType.BINARY_VECTOR, dim=256),
        ]
    )

    schema2.add_function(
        Function(
            name="content_hash",
            function_type=FunctionType.MINHASH,
            input_field_names=["content"],
            output_field_names=["signature"],
            params={"num_hashes": 8, "shingle_size": 5, "token_level": "char"},
        )
    )

    print(f"Schema with function: {schema2.to_dict()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--schema-only":
        # Run schema-only example (no server required)
        example_schema_only()
    else:
        # Run full example (requires Milvus server)
        try:
            main()
        except Exception as e:
            print(f"\nError connecting to Milvus: {e}")
            print("\nTo run schema-only example without Milvus server:")
            print("  python minhash_function_example.py --schema-only")
