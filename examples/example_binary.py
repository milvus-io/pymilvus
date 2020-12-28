"""
This is an example about how to use binary vectors in pymilvus.

Different from normal vectors, each bit of each binary vector represents one dimension.
In python sdk, we use `bytes` object as the structure of binary vectors, which length *
8 could be equal to the dimension value of a binary vector.

You can fina more detailed information about metric and index of binary vectors at
<a href="https://www.milvus.io/docs/metric.md#binary">binary</a>

This example is runnable for Milvus(0.11.x) and pymilvus(0.4.x)(developing).
"""
import random
import struct

import numpy as np

from pprint import pprint

from milvus import Milvus, DataType

# ------
# Setup:
#    Create a client to connect server.
# ------
_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)

# ------
# Basic create collection:
#     Drop all collection.
# ------
collection_name = 'demo_bin_films'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Basic create collection:
#     In our case, the extra infos in "duration" field means the unit of "duration" field is "minutes".
#     And `auto_id` in the parameter is set to `False` so that we can provide our own unique ids.
#     For more information you can refer to the pymilvus
#     documentation (https://pymilvus.readthedocs.io/en/latest/).
# ------
collection_param = {
    "fields": [
        {"name": "duration", "type": DataType.INT32},
        {"name": "release_year", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.BINARY_VECTOR, "params": {"dim": 128}},
    ],
    "segment_row_limit": 100000,
    "auto_id": False
}

# ------
# Basic create collection:
#     After create collection `demo_bin_films`, we create a partition tagged "American", it means the films we
#     will be inserted are from American.
# ------
client.create_collection(collection_name, collection_param)
client.create_partition(collection_name, "American")


# ------
# Basic insert entities:
#     PyMilvus receive `bytes` object as a vector, so we need to generate bytes from list.
#
#     Here we provide two function random binary vectors, which parameters dim refer to dimension of vectors.
# ------
def random_bin_vector(dim):
    """
    This function uses numpy to specify data structure, then convert to bytes
    """
    # uint8 values range is [0, 256), so we specify the high range is 256.
    xb = np.random.randint(256, size=[1, (dim // 8)], dtype="uint8")
    return bytes(xb[0])


def random_bin_vector2(dim):
    """
    This function uses struct.pack to pack as bytes
    """
    rs = [random.randint(0, 255) for _ in range(dim // 8)]
    return struct.pack(f"={dim // 8}B", *rs)


The_Lord_of_the_Rings = [
    {
        "_id": 1,
        "duration": 208,
        "release_year": 2001,
        "embedding": random_bin_vector(128)
    },
    {
        "_id": 2,
        "duration": 226,
        "release_year": 2002,
        "embedding": random_bin_vector(128)
    },
    {
        "_id": 3,
        "duration": 252,
        "release_year": 2003,
        "embedding": random_bin_vector(128)
    }
]

ids = client.insert(collection_name, The_Lord_of_the_Rings, partition_tag="American")
client.flush([collection_name])
print("\n----------insert----------")
print("Films are inserted and the ids are: {}".format(ids))


# ------
# Basic collection stats:
#     We can get the detail of collection statistics info by `get_collection_stats`
# ------
stats = client.get_collection_stats(collection_name)
print("\n----------get collection stats----------")
pprint(stats)


# ------
# Basic hybrid search entities:
#      We are going to get films based on vector similarities.
#      Let's say we have a film with its `embedding` and we want to find `top3` films that are most similar
#      with it by L2 distance.
#      Other than vector similarities, we also want to obtain films that:
#        `released year` term in 2002 or 2003,
#        `duration` larger than 250 minutes.
#
#      Milvus provides Query DSL(Domain Specific Language) to support structured data filtering in queries.
#      For now milvus supports TermQuery and RangeQuery, they are structured as below.
#      For more information about the meaning and other options about "must" and "bool",
#      please refer to DSL chapter of our pymilvus documentation
#      (https://pymilvus.readthedocs.io/en/latest/).
# ------
query_embedding = random_bin_vector2(128)
dsl = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 2003]}
            },
            {
                # "GT" for greater than
                "range": {"duration": {"GT": 250}}
            },
            {
                "vector": {
                    "embedding": {"topk": 3, "query": [query_embedding], "metric_type": "HAMMING"}
                }
            }
        ]
    }
}

# ------
# Basic hybrid search entities:
#     And we want to get all the fields back in results, so fields = ["duration", "release_year", "embedding"].
#     If searching successfully, results will be returned.
#     `results` have `nq`(number of queries) separate results, since we only query for 1 film, The length of
#     `results` is 1.
#     We ask for top 3 in-return, but our condition is too strict while the database is too small, so we can
#     only get 1 film, which means length of `entities` in below is also 1.
#
#     Now we've gotten the results, and known it's a 1 x 1 structure, how can we get ids, distances and fields?
#     It's very simple, for every `topk_film`, it has three properties: `id, distance and entity`.
#     All fields are stored in `entity`, so you can finally obtain these data as below:
#     And the result should be film with id = 3.
# ------
results = client.search(collection_name, dsl, fields=["duration", "release_year", "embedding"])
print("\n----------search----------")
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("- id: {}".format(topk_film.id))
        print("- distance: {}".format(topk_film.distance))

        print("- release_year: {}".format(current_entity.release_year))
        print("- duration: {}".format(current_entity.duration))
        print("- embedding: {}".format(current_entity.embedding))

# ------
# Basic delete:
#     Now let's see how to delete things in Milvus.
#     You can simply delete entities by their ids.
# ------
client.delete_entity_by_id(collection_name, ids=[1, 4])
client.flush()  # flush is important
result = client.get_entity_by_id(collection_name, ids=[1, 4])

counts_delete = sum([1 for entity in result if entity is not None])
counts_in_collection = client.count_entities(collection_name)
print("\n----------delete id = 1, id = 4----------")
print("Get {} entities by id 1, 4".format(counts_delete))
print("There are {} entities after delete films with 1, 4".format(counts_in_collection))

# ------
# Basic delete:
#     You can drop partitions we create, and drop the collection we create.
# ------
client.drop_partition(collection_name, partition_tag='American')
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Summary:
#     Now we've went through all basic communications pymilvus can do with Milvus server, hope it's helpful!
# ------
