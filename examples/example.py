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
#     `auto_id` in the parameter is set to false so that we can provide our own unique ids.
#     `embedding` in the `fields` is float vector with dimension of 8.
#     For more information you can refer to the pymilvus documentation.
# ------
collection_param = {
    "fields": [
        #  Milvus doesn't support String data type now, but we are considering supporting it in the future.
        #  {"name": "film_name", "type": DataType.STRING},
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
#     After insert entities into collection, we need to flush collection to make sure its on disk,
#     so that we are able to retrive it.
# ------
before_flush_counts = client.count_entities(collection_name)
client.flush([collection_name])
after_flush_counts = client.count_entities(collection_name)
print("There are {} films in collection `{}` before flush".format(before_flush_counts, collection_name))
print("There are {} films in collection `{}` after flush".format(after_flush_counts, collection_name))

# ------
# Basic insert entities:
#     We can get the detail of collection statistics info by `get_collection_stats`
# ------
info = client.get_collection_stats(collection_name)
pprint(info)

# ------
# Basic search entities:
#     Now that we have 3 films inserted into our collection, it's time to obtain them.
#     We can get films by ids, if milvus can't find entity for a given id, `None` will be returned.
#     In the case we provide below, we will only get 1 film with id=1 and the other is `None`
# ------
films = client.get_entity_by_id(collection_name, ids=[1, 200])
for film in films:
    if film is not None:
        print(" > id: {},\n > duration: {}m,\n > release_years: {},\n > embedding: {}"
              .format(film.id, film.duration, film.release_year, film.embedding))

# ------
# Basic hybrid search entities:
#      Getting films by id is not enough, we are going to get films based on vector similarities.
#      Let's say we have a film with its `embedding` and we want to find `top3` films that are most similar
#      with it. And there are some conditions for the results. We want to obtain films that are:
#      `released in year` 2002 or 2003,
#      `duration` of the films larger than 250 minutes.
# ------
query_embedding = [random.random() for _ in range(8)]
query_hybrid = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 2003]}
            },
            {
                "range": {"duration": {"GT": 250}}
            },
            {
                "vector": {
                    "embedding": {"topk": 3, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
}

# ------
# Basic hybrid search entities:
#     And we want to get all the fields back in reasults, so fields = ["duration", "release_year", "embedding"]
#     If searching successfully, results will be returned.
#     `results` have `nq`(number of queries) seperate results, since we only query for 1 film, The length of
#     `results` is 1.
#     We ask for top 3 in-return, but our condition is too strict while the database is too small, so we can
#     only get 1 film, which means length of `entities` in below is also 1.
#
#     Now we've gotten the results, and known it's a 1 x 1 structure, how can we get ids, distances and fields?
#     It's very simple, for every `topk_film`, it has three properties: `id, distance and entity`.
#     All fields are stored in `entity`, so you can finally obtain these data as below:
#     And the result should be film with id = 3.
# ------
results = client.search(collection_name, query_hybrid, fields=["duration", "release_year", "embedding"])
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("==")
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
client.delete_entity_by_id(collection_name, ids=[1, 2])
client.flush()  # flush is important
result = client.get_entity_by_id(collection_name, ids=[1, 2])

counts_delete = sum([1 for entity in result if entity is not None])
counts_in_collection = client.count_entities(collection_name)
print("Get {} entities by id 1, 2".format(counts_delete))
print("There are {} entities after delete films with 1, 2".format(counts_in_collection))

# ------
# Basic delete:
#     You can drop partitions we create, and drop the collection we create.
# ------
client.drop_partition(collection_name)
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Summary:
#     Now we've went through all basic communications pymilvus can do with Milvus server, hope it's helpful!
# ------
