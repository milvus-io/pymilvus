# This program demos how to connect to Milvus vector database

import random
from pprint import pprint

from milvus import Milvus, DataType

# ------
# Setup:
#    First of all, you need a runing Milvus. By default, Milvus runs on localhost in port 19530.
#    Then, you can use pymilvus to connect to the server, You can change the _HOST and _PORT accordingly.
# ------
_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)

# ------
# Basic create collection:
#     You already have a Milvus instance running, and pymilvus connecting to Milvus.
#     The first thing we will do is to create a collection `demo_films`. Incase we've already had a collection
#     named `demo_films`, we drop it before we create.
# ------
collection_name = 'demo_films'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Basic create collection:
#     `auto_id` in the parameter is set to true so that we can provide our own unique ids.
#     `embedding` in the `fields` is float vector with dimension of 8.
#     For more information you can refer to the pymilvus documentation.
# ------
collection_param = {
    "fields": [
        {"name": "duration", "type": DataType.INT32, "params": {"unit": "minute"}},
        {"name": "release_year", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
    ],
    "segment_row_limit": 4096,
    "auto_id": False
}

# ------
# Basic create collection:
#     After create collection `demo_films`, we create a partition tagged "American", it means the films we
#     will be inserted are from American.
# ------
client.create_collection(collection_name, collection_param)
client.create_partition(collection_name, "American")

# ------
# Basic create collection:
#     You can check the collection info and partitions we've created by `get_collection_info` and
#     `list_partitions`
# ------
collection = client.get_collection_info(collection_name)
pprint(collection)
partitions = client.list_partitions(collection_name)
pprint(partitions)

# ------
# Basic insert entities:
#     We have three films of The_Lord_of_the_Rings serises here with their id, duration release_year
#     and fake embeddings to be inserted. They are listed below to give you a overview of the structure.
# ------
The_Lord_of_the_Rings = [
    {
        "film": "The_Fellowship_of_the_Ring",
        "id": 1,
        "duration": 208,
        "release_year": 2001,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "film": "The_Two_Towers",
        "id": 2,
        "duration": 226,
        "release_year": 2002,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "film": "The_Return_of_the_King",
        "id": 3,
        "duration": 252,
        "release_year": 2003,
        "embedding": [random.random() for _ in range(8)]
    }
]

# ------
# Basic insert entities:
#     To insert these films into Milvus, we have to group values from the same field together like below.
#     Then these grouped data are used to create `hybrid_entities`.
# ------
ids = [k.get("id") for k in The_Lord_of_the_Rings]
durations = [k.get("duration") for k in The_Lord_of_the_Rings]
release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

hybrid_entities = [
    {"name": "duration", "values": durations, "type": DataType.INT32},
    {"name": "release_year", "values": release_years, "type": DataType.INT32},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
]

# ------
# Basic insert entities:
#     We insert the `hybrid_entities` into our collection, into partition `American`, with ids we provide.
#     If succeed, ids we provide will be returned.
# ------
ids = client.insert(collection_name, hybrid_entities, ids, partition_tag="American")
print("Films are inserted and the ids are: {}".format(ids))


# ------
# Basic insert entities:
#     After insert entities into collection, we need to flush collection to make sure its on disk
#     so that we are able to retrive it.
# ------
before_flush_counts = client.count_entities(collection_name)
client.flush([collection_name])
after_flush_counts = client.count_entities(collection_name)
print("There are {} films in collection `{}` before flush".format(before_flush_counts, collection_name))
print("There are {} films in collection `{}` after flush".format(after_flush_counts, collection_name))

# present collection statistics info
info = client.get_collection_stats(collection_name)
pprint(info)

# Obtain entities by providing ids
films = client.get_entity_by_id(collection_name, ids=[1, 2])
for film in films:
    print(" > id: {}, duration: {}m, release_years: {}, embedding: {}"
          .format(film.id, film.duration, film.release_year, film.embedding))

# search
vectors = [[random.random() for _ in range(8)], ]
query_hybrid = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 2003]}
            },
            {
                "range": {"duration": {"GT": 0}}
            },
            {
                "vector": {
                    "embedding": {"topk": 3, "query": vectors, "metric_type": "L2"}
                }
            }
        ]
    }
}

results = client.search(collection_name, query_hybrid, fields=["duration", "release_year", "embedding"])
print(results)
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("==")
        print("- id: {}".format(topk_film.id))
        print("- distance: {}".format(topk_film.distance))

        print("- release_year: {}".format(current_entity.release_year))
        print("- duration: {}".format(current_entity.duration))
        print("- embedding: {}".format(current_entity.embedding))

client.delete_entity_by_id(collection_name, ids=[1, 2])
# flush is important
client.flush()
result = client.get_entity_by_id(collection_name, ids=[1, 2])
counts = sum([1 for entity in result if entity is not None])
print("Get {} entities by id 1, 2".format(counts))
counts = client.count_entities(collection_name)
print("There are {} entities in the collection".format(counts))


#  # create index of vectors,search more rapidly
#  index_param =   {
#      'nlist': 2048
#  }
#
#  # Create ivflat index in demo_collection
#  # You can search vectors without creating index. however, Creating index help to
#  # search faster
#  print("Creating index: {}".format(index_param))
#  status = client.create_index(collection_name, IndexType.IVF_FLAT, index_param)
#
#  # describe index, get information of index
#  status, index = client.get_index_info(collection_name)
#  print(index)
#
#  # Use the top 10 vectors for similarity search
#  query_vectors = vectors[0:10]
#
#  # execute vector similarity search
#  search_param = {
#      "nprobe": 16
#  }
#
#  print("Searching ... ")
#
#  param = {
#      'collection_name': collection_name,
#      'query_records': query_vectors,
#      'top_k': 1,
#      'params': search_param,
#  }
#
#  status, results = client.search(**param)
#  if status.OK():
#      # indicate search result
#      # also use by:
#      #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
#      if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
#          print('Query result is correct')
#      else:
#          print('Query result isn\'t correct')
#
#      # print results
#      print(results)
#  else:
#      print("Search failed. ", status)
#
#  # Delete demo_collection
#  status = client.drop_collection(collection_name)
if collection_name in client.list_collections():
    client.drop_collection(collection_name)
