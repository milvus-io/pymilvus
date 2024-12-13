import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema(auto_id=False, description="hello_milvus is the simplest demo to introduce the APIs")
schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field("random", DataType.DOUBLE)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", index_type = "IVF_FLAT", metric_type="L2", nlist=128)

print(fmt.format("Create collection `hello_milvus`"))

milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")


print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
]

rows = [ {"pk": entities[0][i], "random": entities[1][i], "embeddings": entities[2][i]} for i in range (num_entities)]

insert_result = milvus_client.insert(collection_name, rows)


print(fmt.format("Start loading"))
milvus_client.load_collection(collection_name)

field_name = "embeddings"

req_list = []
nq = 1
default_limit = 5
vectors_to_search = []

filters = {
    "pk == {str}": {"str": "10"},
    "pk in {list}": {"list": ["1", "10", "100"]},
    "random > {target}": {"target": 5},
    "random <= {target}": {"target": 111.5},
    "{min} <= random < {max}": {"min": 0, "max": 9999},
}

search_param = {
    "data": vectors_to_search,
    "anns_field": field_name,
    "param": {"metric_type": "L2"},
    "limit": default_limit}
vectors_to_search = rng.random((nq, dim))

for filter, filter_params in filters.items():
    print(f"search with filter: {filter}")
    result = milvus_client.search(collection_name=collection_name, data=vectors_to_search, filter=filter, limit=3,
                                  output_fields=["random"], search_params=search_param, filter_params=filter_params)

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}")

    query_results = milvus_client.query(collection_name, filter=filter, output_fields=["random"],
                                        filter_params=filter_params, limit=3)
    for ret in query_results:
        print("query result: ", ret)

print(fmt.format("Search after delete"))
ids = insert_result["ids"]
filter = "pk in {list}"
filter_params = {"list": [ids[0], ids[1]]}
print(f"Start deleting with filter `{filter}`")

result = milvus_client.query(collection_name, filter=filter, output_fields=["random"],
                             filter_params=filter_params, limit=3)
print(f"query before delete by filter=`{filter}` -> result: \n-{result[0]}\n-{result[1]}\n")

milvus_client.delete(collection_name=collection_name, filter=filter, filter_params=filter_params)

result = milvus_client.query(collection_name, filter=filter, output_fields=["random"],
                             filter_params=filter_params, limit=3)
print(f"query after delete by filter=`{filter}` -> result: {result}\n")

print(fmt.format("Release collection"))
milvus_client.release_collection(collection_name)

print(fmt.format("Drop collection"))
milvus_client.drop_collection(collection_name)
