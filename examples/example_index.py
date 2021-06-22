"""
This is an example of creating index

We will be using films.csv file, You can obtain it from here
(https://raw.githubusercontent.com/milvus-io/pymilvus/0.3.0/examples/films.csv)
There are 4 coloumns in films.csv, they are `id`, `title`, `release_year` and `embedding`, all
the data are from MovieLens `ml-latest-small` data except id and embedding, those two columns are fake values.

We will be using `films.csv` to demenstrate how can we build index and search by index on Milvus.
We assuming you have read `example.py` and have a basic conception about how to communicate with Milvus using
pymilvus

This example is runable for Milvus(0.11.x) and pymilvus(0.3.x).
"""
import random
import csv
from pprint import pprint

from pymilvus import Milvus, DataType


_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)

collection_name = 'demo_index'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

collection_param = {
    "fields": [
        {"name": "release_year", "type": DataType.INT64, "is_primary": True},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
    ],
}

client.create_collection(collection_name, collection_param)


# ------
# Basic create index:
#     Now that we have a collection in Milvus with `segment_row_limit` 4096, we can create index or
#     insert entities.
#
#     We can execute `create_index` BEFORE we insert any entites or AFTER. However Milvus won't actually
#     start build index task if the segment row count is smaller than `segment_row_limit`. So if we want
#     to make Milvus build index right away, we need to insert number of entities larger than
#     `segment_row_limit`
#
#     We are going to use data in `films.csv` so you can checkout the structure. And we need to group
#     data with same fields together, so here is a example of how we obtain the data in files and transfer
#     them into what we need.
# ------

ids = []  # ids
titles = []  # titles
release_years = []  # release year
embeddings = []  # embeddings
films = []
with open('films.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    films = [film for film in reader]

for film in films:
    ids.append(int(film[0]))
    titles.append(film[1])
    release_years.append(int(film[2]))
    embeddings.append(list(map(float, film[3][1:][:-1].split(','))))


hybrid_entities = [
    {"name": "release_year", "values": release_years, "type": DataType.INT64},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
]


# ------
# Basic insert:
#     After preparing the data, we are going to insert them into our collection.
#     The number of films inserted should be 8657.
# ------
ids = client.insert(collection_name, hybrid_entities, ids)

client.flush([collection_name])
after_flush_counts = client.count_entities(collection_name)
print(" > There are {} films in collection `{}` after flush".format(after_flush_counts, collection_name))


# ------
# Basic create index:
#     Now that we have insert all the films into Milvus, we are going to build index with these datas.
#
#     While build index, we have to indicate which `field` to build index for, the `index_type`,
#     `metric_type` and params for the specific index type. In our case, we want to build a `IVF_FLAT`
#     index, so the specific params are "nlist". See pymilvus documentation
#     (https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html) for `index_type` we
#     support and the params accordingly
#
#     If there are already index for a collection and you run `create_index` with different params the
#     older index will be replaced by new one.
# ------
client.create_index(collection_name, "embedding",
                    {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})

# ------
# Basic create index:
#     We can get the detail of the index  by `describe_collection`
# ------
info = client.describe_collection(collection_name)
pprint(info)

# ------
# Basic hybrid search entities:
#     If we want to use index, the specific index params need to be provided, in our case, the "params"
#     should be "nprobe", if no "params" given, Milvus will complain about it and raise a exception.
# ------
query_embedding = [random.random() for _ in range(8)]
query_hybrid = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 1995]}
            },
            {
                "vector": {
                    "embedding": {"topk": 3,
                                  "query": [query_embedding],
                                  "metric_type": "L2",
                                  "params": {"nprobe": 8}}
                }
            }
        ]
    }
}

# ------
# Basic hybrid search entities
# ------
results = client.search(collection_name, query_hybrid, fields=["release_year", "embedding"])
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("==")
        print("- id: {}".format(topk_film.id))
        print("- title: {}".format(titles[topk_film.id]))
        print("- distance: {}".format(topk_film.distance))

        print("- release_year: {}".format(current_entity.release_year))
        print("- embedding: {}".format(current_entity.embedding))


# ------
# Basic delete index:
#     You can drop index we create
# ------
client.drop_index(collection_name, "embedding")

if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Summary:
#     Now we've went through some basic build index operations, hope it's helpful!
# ------
