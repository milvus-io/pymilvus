import numpy as np
import time 
import random

from pymilvus import (
    connections,
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

dim = 128
nb = 10000
nq = 10
collection_name = "GPU_CAGRA_test"

# configure milvus hostname and port
print(f"\nCreate connection...")
connections.connect(host="localhost", port=19530)

# List all collection names
print(f"\nList collections...")
collection_list = list_collections()
print(list_collections())

if(collection_list.count(collection_name)):
    print(collection_name, " exist, and drop it")
    collection = Collection(collection_name)
    collection.drop()
    print("drop")

field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True)
field2 = FieldSchema(name = "vec", dtype = DataType.FLOAT_VECTOR, description = "float vector", dim = dim, is_primary = False)
schema = CollectionSchema(fields = [field1, field2], description = "sift decription")
collection = Collection(name = collection_name, data = None, schema = schema, shards_num = 2)

print(list_collections())

print(f"\nList partitions...")
print(collection.partitions)

print("begin insert...")
rng = np.random.default_rng(seed=19530)
data = rng.random((nb, dim))
counter = 0
block_num = 100
block_size = int(data.shape[0]/block_num)
start = time.time()
for t in range(block_num):
    entities = [
            [i for i in range(counter, counter + block_size)],
            # [vectors[i] for i in range(counter, counter + block_size)]
            [vec for vec in data[counter: counter + block_size]]
            ]
    insert_result =  collection.insert(entities)
    counter = counter + block_size
print ("end of insert, cost: ", time.time()-start)

collection.flush()
print(collection.num_entities)

# create index
print(f"\nCreate index...")

collection.create_index(field_name="vec",
        index_params={'index_type': 'GPU_CAGRA',  
            'metric_type': 'L2',
            'params': {
                'intermediate_graph_degree':64,
                'graph_degree': 32,
                }})
print(f"\nCreated index done.")

# load
print(f"\nLoad...")
collection.load()
print(f"\nLoaded")

print(f"\nSearch...")
res = collection.search([ vec for vec in data[0:1]],
                        "vec", 
                            {"metric_type": "L2",
                              "params": {
                                   "search_width":100},
                                   }, 
                                limit=100)
print("run result: ", res[0].ids)
collection.drop()
