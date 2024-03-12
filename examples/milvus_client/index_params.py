from pymilvus.milvus_client import IndexParams



index_params = IndexParams()

index_params.add_index(field_name = "embeddings", metric_type="L2", efConstruction=100, M=10)
# The operation of this line of code will supersede the operation of the previous line
index_params.add_index(field_name = "embeddings", metric_type="L2", params = {"efConstruction":100, "M":20})
index_params.add_index(field_name = "title", index_type = "Trie", index_name="my_trie")
index_params.add_index(field_name = "text", index_type = "INVERTED", index_name="my_inverted")

for i in index_params:
    print(i)

