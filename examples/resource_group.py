from pymilvus import (
    MilvusClient,
    DataType,
)
from pymilvus.client.constants import DEFAULT_RESOURCE_GROUP

from pymilvus.client.types import (
    ResourceGroupConfig,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")


## create collection and load collection
print("create collection and load collection")
collection_name = "hello_milvus"
has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema(enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("title", DataType.VARCHAR, max_length=64)
milvus_client.create_collection(collection_name, schema=schema, consistency_level="Strong")
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", metric_type="L2")
index_params.add_index(field_name = "title", index_type = "Trie", index_name="my_trie")
milvus_client.create_index(collection_name, index_params)
milvus_client.load_collection(collection_name)


## create resource group
print("create resource group")
milvus_client.create_resource_group("rg1")
milvus_client.create_resource_group("rg2")

## update resource group
configs = {
            "rg1": ResourceGroupConfig(
                requests={"node_num": 1},
                limits={"node_num": 5},
                transfer_from=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
                transfer_to=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
            ),
            "rg2": ResourceGroupConfig(
                requests={"node_num": 4},
                limits={"node_num": 4},
                transfer_from=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
                transfer_to=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
            ),
        }
milvus_client.update_resource_groups(configs)

## describe resource group
print("describe rg1")
result = milvus_client.describe_resource_group("rg1")
print(result)

print("describe rg2")
result = milvus_client.describe_resource_group("rg2")
print(result)

## list resource group
print("list resource group")
result = milvus_client.list_resource_groups()
print(result)

## transfer replica
print("transfer replica to rg1")
milvus_client.transfer_replica(DEFAULT_RESOURCE_GROUP, "rg1", collection_name, 1)
print("describe rg1 after transfer replica in")
result = milvus_client.describe_resource_group("rg1")
print(result)

milvus_client.transfer_replica("rg1", DEFAULT_RESOURCE_GROUP, collection_name, 1)
print("describe rg1 after transfer replica out")
result = milvus_client.describe_resource_group("rg1")
print(result)

## drop resource group
print("drop resource group")
# create resource group
configs = {
            "rg1": ResourceGroupConfig(
                requests={"node_num": 0},
                limits={"node_num": 0},
                transfer_from=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
                transfer_to=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
            ),
            "rg2": ResourceGroupConfig(
                requests={"node_num": 0},
                limits={"node_num": 0},
                transfer_from=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
                transfer_to=[{"resource_group": DEFAULT_RESOURCE_GROUP}],
            ),
        }
milvus_client.update_resource_groups(configs)
milvus_client.drop_resource_group("rg1")
milvus_client.drop_resource_group("rg2")


