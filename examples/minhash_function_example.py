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

=============================================================================
IMPORTANT CONSTRAINTS:
- Index Type: MUST use MINHASH_LSH (only supported index for MinHash output)
- Metric Type: MUST use MHJACCARD (enforced by server)
- Output Field: MUST be BINARY_VECTOR with dim = num_hashes * 32
=============================================================================

MinHash Function Parameters:
----------------------------
| Parameter      | Type   | Default   | Description                           |
|----------------|--------|-----------|---------------------------------------|
| num_hashes     | int    | dim/32    | Number of hash functions              |
| shingle_size   | int    | 3         | N-gram size for text shingling        |
| hash_function  | string | xxhash64  | Hash function: "xxhash64" or "sha1"   |
| token_level    | string | word      | Tokenization: "word" or "char"        |
| seed           | int    | 1234      | Random seed for reproducibility       |

Index Parameters (MINHASH_LSH):
-------------------------------
| Parameter                       | Type  | Required | Description                    |
|---------------------------------|-------|----------|--------------------------------|
| mh_lsh_band                     | int   | Yes      | LSH band count                 |
| mh_element_bit_width            | int   | No       | Element bit width              |
| mh_lsh_code_in_mem              | int   | No       | LSH code in memory (0 or 1)    |
| with_raw_data                   | bool  | No       | Keep raw data for reranking    |
| mh_lsh_bloom_false_positive_prob| float | No       | Bloom filter FP probability    |

Search Parameters:
------------------
| Parameter              | Type | Description                              |
|------------------------|------|------------------------------------------|
| mh_search_with_jaccard | bool | Use exact Jaccard for reranking          |
| refine_k               | int  | Candidate count for refinement stage     |
| mh_lsh_batch_search    | bool | Enable batch search mode                 |
"""

from pymilvus import MilvusClient, DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema, Function
from pymilvus.client.types import FunctionType


# =============================================================================
# MinHash Function Parameters Reference
# =============================================================================

# All available MinHash function parameters with descriptions
MINHASH_FUNCTION_PARAMS = {
    "num_hashes": 16,           # Number of hash functions (output dim = num_hashes * 32)
    "shingle_size": 3,          # N-gram window size (larger = more context-aware)
    "hash_function": "xxhash64",  # "xxhash64" (faster) or "sha1" (cryptographic)
    "token_level": "word",      # "word" (semantic) or "char" (character-level)
    "seed": 1234,               # Random seed for reproducibility
}

# All available MINHASH_LSH index parameters
MINHASH_INDEX_PARAMS = {
    "mh_lsh_band": 8,                        # Required: LSH band count (higher = better recall, slower)
    "mh_element_bit_width": 32,              # Optional: Element bit width
    "mh_lsh_code_in_mem": 1,                 # Optional: Store LSH codes in memory (1=yes, 0=no)
    "with_raw_data": True,                   # Optional: Keep raw data for exact reranking
    "mh_lsh_bloom_false_positive_prob": 0.01,  # Optional: Bloom filter false positive probability
}

# All available search parameters for MinHash
MINHASH_SEARCH_PARAMS = {
    "mh_search_with_jaccard": True,   # Use exact Jaccard distance for reranking
    "refine_k": 50,                   # Number of candidates to refine
    "mh_lsh_batch_search": True,      # Enable batch search mode for better throughput
}


# =============================================================================
# Example 1: Basic Usage with Default Parameters
# =============================================================================

def create_basic_minhash_collection(client: MilvusClient, collection_name: str):
    """Create a collection with MinHash function using basic/default parameters.

    This is the simplest way to use MinHash for text deduplication.
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=512)

    # Basic MinHash function with minimal parameters
    schema.add_function(Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            "num_hashes": 16,      # dim = 16 * 32 = 512
            "shingle_size": 3,
        },
    ))

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created basic collection: {collection_name}")
    return schema


def create_basic_index(client: MilvusClient, collection_name: str):
    """Create a basic MINHASH_LSH index with minimal parameters."""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="minhash_signature",
        index_type="MINHASH_LSH",      # REQUIRED: Only supported index type
        metric_type="MHJACCARD",        # REQUIRED: Only supported metric type
        params={
            "mh_lsh_band": 8,            # Required parameter
        },
    )
    client.create_index(collection_name, index_params)
    print(f"Created basic MINHASH_LSH index")


# =============================================================================
# Example 2: Advanced Usage with All Parameters
# =============================================================================

def create_advanced_minhash_collection(client: MilvusClient, collection_name: str):
    """Create a collection with MinHash function using ALL available parameters.

    This demonstrates the full configuration options for fine-tuning MinHash behavior.
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=1024)

    # MinHash function with ALL parameters
    schema.add_function(Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            # Number of hash functions: determines signature granularity
            # Higher = more accurate but larger vectors
            # dim = num_hashes * 32
            "num_hashes": 32,  # dim = 32 * 32 = 1024

            # N-gram (shingle) size: affects similarity sensitivity
            # Smaller = sensitive to minor changes
            # Larger = focuses on overall structure
            "shingle_size": 5,

            # Hash function choice:
            # - "xxhash64": Fast, good for most use cases (default)
            # - "sha1": Cryptographic, slower but more uniform distribution
            "hash_function": "xxhash64",

            # Tokenization level:
            # - "word": Word-level shingling (better for natural language)
            # - "char": Character-level shingling (better for short text, typos)
            "token_level": "word",

            # Random seed for reproducibility
            # Same seed = same hash permutations = reproducible results
            "seed": 42,
        },
    ))

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created advanced collection: {collection_name}")
    return schema


def create_advanced_index(client: MilvusClient, collection_name: str):
    """Create a MINHASH_LSH index with ALL available parameters.

    IMPORTANT:
    - index_type MUST be "MINHASH_LSH"
    - metric_type MUST be "MHJACCARD"
    """
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="minhash_signature",
        index_type="MINHASH_LSH",      # REQUIRED: Only supported index type
        metric_type="MHJACCARD",        # REQUIRED: Only supported metric type
        params={
            # Required: Number of LSH bands
            # Higher = better recall but slower search
            # Typical range: 4-16
            "mh_lsh_band": 12,

            # Optional: Element bit width (default: 32)
            "mh_element_bit_width": 32,

            # Optional: Store LSH codes in memory
            # 1 = in memory (faster search)
            # 0 = on disk (lower memory usage)
            "mh_lsh_code_in_mem": 1,

            # Optional: Keep raw MinHash data for exact Jaccard reranking
            # True = enables mh_search_with_jaccard for accurate results
            # False = LSH approximation only (faster but less accurate)
            "with_raw_data": True,

            # Optional: Bloom filter false positive probability
            # Lower = more accurate filtering but higher memory
            # Typical range: 0.001 - 0.1
            "mh_lsh_bloom_false_positive_prob": 0.01,
        },
    )
    client.create_index(collection_name, index_params)
    print(f"Created advanced MINHASH_LSH index with all parameters")


# =============================================================================
# Example 3: Different Search Strategies
# =============================================================================

def search_with_basic_params(client: MilvusClient, collection_name: str, query: str, limit: int = 5):
    """Basic search with minimal parameters."""
    return client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="minhash_signature",
        search_params={
            "metric_type": "MHJACCARD",
            "params": {},  # Use defaults
        },
        limit=limit,
        output_fields=["id", "text"],
    )


def search_with_jaccard_reranking(client: MilvusClient, collection_name: str, query: str, limit: int = 5):
    """Search with exact Jaccard distance reranking for higher accuracy.

    Requires index to be built with `with_raw_data=True`.
    """
    return client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="minhash_signature",
        search_params={
            "metric_type": "MHJACCARD",
            "params": {
                # Enable exact Jaccard distance calculation for reranking
                # This improves accuracy but is slower
                "mh_search_with_jaccard": True,

                # Number of candidates to retrieve before reranking
                # Higher = better recall but slower
                "refine_k": 100,
            },
        },
        limit=limit,
        output_fields=["id", "text"],
    )


def search_with_batch_mode(client: MilvusClient, collection_name: str, queries: list, limit: int = 5):
    """Batch search for multiple queries with optimized throughput.

    Use this when searching many queries at once.
    """
    return client.search(
        collection_name=collection_name,
        data=queries,  # Multiple queries
        anns_field="minhash_signature",
        search_params={
            "metric_type": "MHJACCARD",
            "params": {
                # Enable batch search mode for better throughput
                "mh_lsh_batch_search": True,

                # Can combine with Jaccard reranking
                "mh_search_with_jaccard": True,
                "refine_k": 50,
            },
        },
        limit=limit,
        output_fields=["id", "text"],
    )


def search_with_all_params(client: MilvusClient, collection_name: str, query: str, limit: int = 5):
    """Search using ALL available search parameters."""
    return client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="minhash_signature",
        search_params={
            "metric_type": "MHJACCARD",  # Required
            "params": {
                # Exact Jaccard reranking (requires with_raw_data=True in index)
                "mh_search_with_jaccard": True,

                # Refine candidate count
                "refine_k": 100,

                # Batch search mode for throughput optimization
                "mh_lsh_batch_search": True,
            },
        },
        limit=limit,
        output_fields=["id", "text"],
    )


# =============================================================================
# Example 4: Character-level vs Word-level MinHash
# =============================================================================

def create_char_level_minhash_collection(client: MilvusClient, collection_name: str):
    """Create collection with character-level MinHash.

    Character-level MinHash is better for:
    - Short texts (titles, names)
    - Typo detection
    - Non-word-boundary languages (Chinese, Japanese)
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=512)

    schema.add_function(Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            "num_hashes": 16,
            "shingle_size": 3,       # 3-character n-grams
            "token_level": "char",   # Character-level tokenization
            "hash_function": "xxhash64",
        },
    ))

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created char-level MinHash collection: {collection_name}")
    return schema


def create_word_level_minhash_collection(client: MilvusClient, collection_name: str):
    """Create collection with word-level MinHash.

    Word-level MinHash is better for:
    - Long documents
    - Semantic similarity
    - Languages with clear word boundaries
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=512)

    schema.add_function(Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            "num_hashes": 16,
            "shingle_size": 3,       # 3-word n-grams
            "token_level": "word",   # Word-level tokenization
            "hash_function": "xxhash64",
        },
    ))

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created word-level MinHash collection: {collection_name}")
    return schema


# =============================================================================
# Example 5: SHA1 vs xxHash64 Hash Functions
# =============================================================================

def create_sha1_minhash_collection(client: MilvusClient, collection_name: str):
    """Create collection with SHA1 hash function.

    SHA1 is:
    - Cryptographically stronger
    - More uniform distribution
    - Slower than xxhash64
    - Good for security-sensitive applications
    """
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="minhash_signature", datatype=DataType.BINARY_VECTOR, dim=512)

    schema.add_function(Function(
        name="text_to_minhash",
        function_type=FunctionType.MINHASH,
        input_field_names=["text"],
        output_field_names=["minhash_signature"],
        params={
            "num_hashes": 16,
            "shingle_size": 3,
            "hash_function": "sha1",  # Cryptographic hash
            "token_level": "word",
        },
    ))

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created SHA1 MinHash collection: {collection_name}")
    return schema


# =============================================================================
# Main Demo Functions
# =============================================================================

def insert_sample_documents(client: MilvusClient, collection_name: str):
    """Insert sample documents for testing."""
    documents = [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
        {"id": 2, "text": "A quick brown fox jumped over a lazy dog."},
        {"id": 3, "text": "The fast brown fox leaps over the sleepy dog."},
        {"id": 4, "text": "Machine learning is transforming artificial intelligence."},
        {"id": 5, "text": "Deep learning transforms artificial intelligence research."},
        {"id": 6, "text": "Python is a popular programming language for data science."},
        {"id": 7, "text": "Data science uses Python as a popular language."},
        {"id": 8, "text": "The weather today is sunny and warm."},
        {"id": 9, "text": "Today's weather is warm and sunny."},
        {"id": 10, "text": "Completely unrelated text about cooking recipes."},
    ]
    client.insert(collection_name, documents)
    client.flush(collection_name)
    print(f"Inserted {len(documents)} documents")


def demo_basic_usage(uri: str):
    """Demonstrate basic MinHash usage."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic MinHash Usage")
    print("=" * 70)

    client = MilvusClient(uri=uri)
    collection_name = "minhash_basic_demo"

    # Create and setup
    create_basic_minhash_collection(client, collection_name)
    create_basic_index(client, collection_name)
    insert_sample_documents(client, collection_name)
    client.load_collection(collection_name)

    # Search
    query = "The fast brown fox jumps over the lazy dog."
    print(f"\nQuery: '{query}'")
    results = search_with_basic_params(client, collection_name, query)

    print("\nResults (basic search):")
    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. ID={hit['id']}, Distance={hit['distance']:.4f}")
        print(f"     Text: {hit['entity']['text']}")

    client.drop_collection(collection_name)


def demo_advanced_usage(uri: str):
    """Demonstrate advanced MinHash usage with all parameters."""
    print("\n" + "=" * 70)
    print("DEMO 2: Advanced MinHash Usage (All Parameters)")
    print("=" * 70)

    client = MilvusClient(uri=uri)
    collection_name = "minhash_advanced_demo"

    # Create with all parameters
    create_advanced_minhash_collection(client, collection_name)
    create_advanced_index(client, collection_name)
    insert_sample_documents(client, collection_name)
    client.load_collection(collection_name)

    # Search with Jaccard reranking
    query = "Machine learning and artificial intelligence"
    print(f"\nQuery: '{query}'")

    print("\n--- Search with Jaccard Reranking ---")
    results = search_with_jaccard_reranking(client, collection_name, query)
    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. ID={hit['id']}, Distance={hit['distance']:.4f}")
        print(f"     Text: {hit['entity']['text']}")

    print("\n--- Batch Search ---")
    queries = [
        "Python programming language",
        "Weather today sunny",
        "Brown fox jumping",
    ]
    print(f"Queries: {queries}")
    batch_results = search_with_batch_mode(client, collection_name, queries, limit=2)
    for q_idx, (query_text, result) in enumerate(zip(queries, batch_results)):
        print(f"\n  Query {q_idx + 1}: '{query_text}'")
        for hit in result:
            print(f"    - ID={hit['id']}, Distance={hit['distance']:.4f}: {hit['entity']['text'][:40]}...")

    client.drop_collection(collection_name)


def demo_search_params_comparison(uri: str):
    """Compare different search parameter configurations."""
    print("\n" + "=" * 70)
    print("DEMO 3: Search Parameters Comparison")
    print("=" * 70)

    client = MilvusClient(uri=uri)
    collection_name = "minhash_search_params_demo"

    create_advanced_minhash_collection(client, collection_name)
    create_advanced_index(client, collection_name)
    insert_sample_documents(client, collection_name)
    client.load_collection(collection_name)

    query = "The quick brown fox"
    print(f"\nQuery: '{query}'")

    print("\n--- Basic Search (LSH only) ---")
    results1 = search_with_basic_params(client, collection_name, query)
    for hit in results1[0][:3]:
        print(f"  ID={hit['id']}, Distance={hit['distance']:.4f}")

    print("\n--- With Jaccard Reranking (refine_k=100) ---")
    results2 = search_with_jaccard_reranking(client, collection_name, query)
    for hit in results2[0][:3]:
        print(f"  ID={hit['id']}, Distance={hit['distance']:.4f}")

    print("\n--- All Parameters (Jaccard + Batch) ---")
    results3 = search_with_all_params(client, collection_name, query)
    for hit in results3[0][:3]:
        print(f"  ID={hit['id']}, Distance={hit['distance']:.4f}")

    client.drop_collection(collection_name)


def example_schema_only():
    """Example showing schema creation without Milvus server.

    This demonstrates all parameter options without requiring a running server.
    """
    print("=" * 70)
    print("MinHash Schema Definition Examples (No Server Required)")
    print("=" * 70)

    # Example 1: Basic configuration
    print("\n[1] Basic MinHash Configuration:")
    print("-" * 40)
    schema1 = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
        ],
        functions=[
            Function(
                name="text_to_minhash",
                function_type=FunctionType.MINHASH,
                input_field_names=["text"],
                output_field_names=["signature"],
                params={"num_hashes": 16, "shingle_size": 3},
            )
        ],
    )
    print(f"  dim=512, num_hashes=16, shingle_size=3")
    print(f"  Function: {schema1.functions[0].to_dict()}")

    # Example 2: Full configuration
    print("\n[2] Full MinHash Configuration (All Function Params):")
    print("-" * 40)
    schema2 = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("signature", DataType.BINARY_VECTOR, dim=1024),
        ],
        functions=[
            Function(
                name="text_to_minhash",
                function_type=FunctionType.MINHASH,
                input_field_names=["text"],
                output_field_names=["signature"],
                params={
                    "num_hashes": 32,        # dim = 32 * 32 = 1024
                    "shingle_size": 5,
                    "hash_function": "xxhash64",
                    "token_level": "word",
                    "seed": 42,
                },
            )
        ],
    )
    print(f"  dim=1024, num_hashes=32, shingle_size=5")
    print(f"  hash_function=xxhash64, token_level=word, seed=42")
    print(f"  Function: {schema2.functions[0].to_dict()}")

    # Example 3: Character-level
    print("\n[3] Character-level MinHash:")
    print("-" * 40)
    schema3 = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("signature", DataType.BINARY_VECTOR, dim=256),
        ],
        functions=[
            Function(
                name="char_minhash",
                function_type=FunctionType.MINHASH,
                input_field_names=["text"],
                output_field_names=["signature"],
                params={
                    "num_hashes": 8,
                    "shingle_size": 4,
                    "token_level": "char",  # Character-level
                },
            )
        ],
    )
    print(f"  token_level=char, shingle_size=4 (4-character n-grams)")

    # Example 4: SHA1 hash function
    print("\n[4] SHA1 Hash Function:")
    print("-" * 40)
    schema4 = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
        ],
        functions=[
            Function(
                name="sha1_minhash",
                function_type=FunctionType.MINHASH,
                input_field_names=["text"],
                output_field_names=["signature"],
                params={
                    "num_hashes": 16,
                    "shingle_size": 3,
                    "hash_function": "sha1",  # Cryptographic hash
                },
            )
        ],
    )
    print(f"  hash_function=sha1 (cryptographic, slower but uniform)")

    # Print index and search params reference
    print("\n" + "=" * 70)
    print("Index Parameters Reference (MINHASH_LSH):")
    print("=" * 70)
    print("""
    Required:
    - index_type: "MINHASH_LSH"          (Only supported type)
    - metric_type: "MHJACCARD"           (Only supported metric)
    - mh_lsh_band: int                   (LSH band count, e.g., 8)

    Optional:
    - mh_element_bit_width: int          (Element bit width, default: 32)
    - mh_lsh_code_in_mem: int            (1=in memory, 0=on disk)
    - with_raw_data: bool                (Keep raw data for reranking)
    - mh_lsh_bloom_false_positive_prob: float  (Bloom filter FP prob)
    """)

    print("=" * 70)
    print("Search Parameters Reference:")
    print("=" * 70)
    print("""
    - mh_search_with_jaccard: bool       (Use exact Jaccard reranking)
    - refine_k: int                      (Candidate count for refinement)
    - mh_lsh_batch_search: bool          (Enable batch search mode)
    """)


def main():
    """Main function to run all demos."""
    uri = "http://localhost:19530"

    print("=" * 70)
    print("MinHash Function Complete Example")
    print("Covering ALL parameters for Function, Index, and Search")
    print("=" * 70)

    try:
        client = MilvusClient(uri=uri)
        print(f"\nConnected to Milvus: {uri}")
        print(f"Server version: {client.get_server_version()}")

        # Run demos
        demo_basic_usage(uri)
        demo_advanced_usage(uri)
        demo_search_params_comparison(uri)

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTo run schema-only examples without Milvus server:")
        print("  python minhash_function_example.py --schema-only")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--schema-only":
        example_schema_only()
    else:
        main()
