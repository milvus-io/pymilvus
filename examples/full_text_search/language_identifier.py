from pymilvus import MilvusClient, DataType, Function, FunctionType

# 1. Setup Milvus Client
client = MilvusClient("http://localhost:19530")
COLLECTION_NAME = "multilingual_test_B"
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

# --- Option B: Using Language Identifier Tokenizer ---
print("\n--- Demonstrating Language Identifier Tokenizer ---")

# 3B. reate a collection with language identifier
analyzer_params_langid = {
    "tokenizer": {
        "type": "language_identifier",
        "analyzers": analyzers # Referencing the analyzers defined in Step 2
    },
}

schema_langid = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)
schema_langid.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
# The 'language' field is not strictly needed by the analyzer itself here, as detection is automatic.# However, you might keep it for metadata purposes.
schema_langid.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=25565, enable_analyzer=True, analyzer_params = analyzer_params_langid)
schema_langid.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR) # BM25 Sparse Vector# add bm25 function
text_bm25_function_langid = Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["text_sparse"],
)
schema_langid.add_function(text_bm25_function_langid)

index_params_langid = client.prepare_index_params()
index_params_langid.add_index(
    field_name="text_sparse",
    index_type="AUTOINDEX", # Use auto index for BM25
    metric_type="BM25",
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema_langid,
    index_params=index_params_langid
)
print(f"Collection '{COLLECTION_NAME}' created successfully with Language Identifier Tokenizer.")

# 4B. Insert Data for Language Identifier Tokenizer and Load Collection
# Insert English and Japanese movie titles. The language_identifier will detect the language.
client.insert(
    collection_name=COLLECTION_NAME,
    data=[
        {"text": "The Lord of the Rings"},
        {"text": "Spirited Away"},
        {"text": "千と千尋の神隠し"}, 
        {"text": "君の名は。"},
    ]
)
print(f"Inserted multilingual data into '{COLLECTION_NAME}'.")

# Load the collection into memory before searching
client.load_collection(collection_name=COLLECTION_NAME)

# 5B. Perform a full-text search with Language Identifier Tokenizer# No need to specify analyzer_name in search_params; it's detected automatically for the query.
print("\n--- Search results for Language Identifier Tokenizer ---")
results_langid_jp = client.search(
    collection_name=COLLECTION_NAME,
    data=["神隠し"],
    limit=2,
    output_fields=["text"],
    search_params={"metric_type": "BM25"}, # Analyzer automatically determined by language_identifier
    consistency_level = "Strong",
)
print("\nSearch results for '神隠し' (Language Identifier Tokenizer):")
for result in results_langid_jp[0]:
    print(result)

results_langid_en = client.search(
    collection_name=COLLECTION_NAME,
    data=["the Rings"],
    limit=2,
    output_fields=["text"],
    search_params={"metric_type": "BM25"}, # Analyzer automatically determined by language_identifier
    consistency_level = "Strong",
)
print("\nSearch results for 'the Rings' (Language Identifier Tokenizer):")
for result in results_langid_en[0]:
    print(result)

client.drop_collection(collection_name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' dropped.")