from pymilvus import MilvusClient, DataType, Function, FunctionType

# 1. Setup Milvus Client
client = MilvusClient("http://localhost:19530")
COLLECTION_NAME = "multilingual_test_A"
if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)

# 2. Define analyzers for multiple languages
# These individual analyzer definitions will be reused by both methods.
analyzers = {
    "Japanese": { 
        # Use lindera with japanese dict 'ipadic' 
        # and remove punctuation beacuse lindera tokenizer will remain punctuation
        "tokenizer":{
            "type": "lindera",
            "dict_kind": "ipadic"
        },
        "filter": ["removepunct"]
    },
    "English": {
        # Use build-in english analyzer
        "type": "english",
    },
    "default": {
        # use icu tokenizer as a fallback.
        "tokenizer": "icu",
    }
}

# --- Option A: Using Multi-Language Analyzer ---
print("\n--- Demonstrating Multi-Language Analyzer ---")

# 3A. reate a collection with the Multi Analyzer

mutil_analyzer_params = {
    "by_field": "language",
    "analyzers": analyzers,
}

schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)# Apply our multi-language analyzer to the 'title' field
schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=255, nullable = True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=25565, enable_analyzer=True, multi_analyzer_params = mutil_analyzer_params)
schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR) # Bm25 Sparse Vector

# add bm25 function
text_bm25_function = Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["text_sparse"],
)
schema.add_function(text_bm25_function)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="text_sparse",
    index_type="AUTOINDEX", # Use auto index for BM25
    metric_type="BM25",
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)
print(f"Collection '{COLLECTION_NAME}' created successfully.")

# 4A. Insert data for Multi-Language Analyzer and load collection# Insert English and Japanese movie titles, explicitly setting the 'language' field
client.insert(
    collection_name=COLLECTION_NAME,
    data=[
        {"text": "The Lord of the Rings", "language": "English"},
        {"text": "Spirited Away", "language": "English"},
        {"text": "千と千尋の神隠し", "language": "Japanese"}, # This is "Spirited Away" in Japanese
        {"text": "君の名は。", "language": "Japanese"}, # This is "Your Name." in Japanese
    ]
)
print(f"Inserted multilingual data into '{COLLECTION_NAME}'.")

# Load the collection into memory before searching
client.load_collection(collection_name=COLLECTION_NAME)

# 5A. Perform a full-text search with Multi-Language Analyzer# When searching, explicitly specify the analyzer to use for the query string.
print("\n--- Search results for Multi-Language Analyzer ---")
results_multi_jp = client.search(
    collection_name=COLLECTION_NAME,
    data=["神隠し"],
    limit=2,
    output_fields=["text"],
    search_params={"metric_type": "BM25", "analyzer_name": "Japanese"}, # Specify Japanese analyzer for query
    consistency_level = "Strong",
)
print("\nSearch results for '神隠し' (Multi-Language Analyzer):")
for result in results_multi_jp[0]:
    print(result)

results_multi_en = client.search(
    collection_name=COLLECTION_NAME,
    data=["Rings"],
    limit=2,
    output_fields=["text"],
    search_params={"metric_type": "BM25", "analyzer_name": "English"}, # Specify English analyzer for query
    consistency_level = "Strong",
)
print("\nSearch results for 'Rings' (Multi-Language Analyzer):")
for result in results_multi_en[0]:
    print(result)

client.drop_collection(collection_name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' dropped.")