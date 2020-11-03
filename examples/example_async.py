"""
This is an example to demonstrate how to invoke milvus client APIs asynchronously.

There are partial APIs allowed to be invoked asynchronously, they are: insert(), create_index(),
search(), flush() and compact().

This example is runnable for milvus(0.11.x) and pymilvus(0.3.x).
"""
import random
from pprint import pprint

from milvus import Milvus, DataType

# ------
# Setup:
#    First of all, you need a runing Milvus(0.11.x). By default, Milvus runs on localhost in port 19530.
#    Then, you can use pymilvus(0.3.x) to connect to the server, You can change the _HOST and _PORT accordingly.
# ------
_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)

# ------
# Basic create collection:
#     You already have a Milvus instance running, and pymilvus connecting to Milvus.
#     The first thing we will do is to create a collection `demo_films`. In case we've already had a collection
#     named `demo_films`, we drop it before we create.
# ------
collection_name = 'demo_films'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# ------
# Basic create collection:
#     For a specific field, you can provide extra infos by a dictionary with `key = "params"`. If the field
#     has a type of `FLOAT_VECTOR` and `BINARY_VECTOR`, "dim" must be provided in extra infos. Otherwise
#     you can provide customed infos like `{"unit": "minutes"}` for you own need.
#
#     In our case, the extra infos in "duration" field means the unit of "duration" field is "minutes".
#     And `auto_id` in the parameter is set to `False` so that we can provide our own unique ids.
#     For more information you can refer to the pymilvus
#     documentation (https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html).
# ------
collection_param = {
    "fields": [
        #  Milvus doesn't support string type now, but we are considering supporting it soon.
        #  {"name": "title", "type": DataType.STRING},
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
# Basic insert entities:
#     We have two films groups of The_Lord_of_the_Rings series and Batman series here with their id, duration
#     release_year and fake embeddings to be inserted. They are listed below to give you a overview of the structure.
# ------

The_Lord_of_the_Rings = [
    {
        "title": "The_Fellowship_of_the_Ring",
        "id": 1,
        "duration": 208,
        "release_year": 2001,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Two_Towers",
        "id": 2,
        "duration": 226,
        "release_year": 2002,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Return_of_the_King",
        "id": 3,
        "duration": 252,
        "release_year": 2003,
        "embedding": [random.random() for _ in range(8)]
    }
]

Batmans = [
    {
        "title": "Batman_Begins",
        "id": 4,
        "duration": 140,
        "release_year": 2005,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "Batman_The_Dark_Knight",
        "id": 5,
        "duration": 152,
        "release_year": 2008,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "Batman_The_Dark_Knight_Rises",
        "id": 6,
        "duration": 165,
        "release_year": 2012,
        "embedding": [random.random() for _ in range(8)]
    }
]


# ------
# Basic insert entities:
#     To insert two films groups into Milvus, we provide a function to change these groups to object can be passed
#     directly. we have to group values from the same field together like below.
#     Then these grouped data are used to create `hybrid_entities`.
# ------

def columnar_entities(entities):
    ids = [k.get("id") for k in entities]
    durations = [k.get("duration") for k in entities]
    release_years = [k.get("release_year") for k in entities]
    embeddings = [k.get("embedding") for k in entities]

    return ids, [
        # Milvus doesn't support string type yet, so we cannot insert "title".
        {"name": "duration", "values": durations, "type": DataType.INT32},
        {"name": "release_year", "values": release_years, "type": DataType.INT32},
        {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
    ]


rings_ids, rings_entities = columnar_entities(The_Lord_of_the_Rings)
batman_ids, batman_entities = columnar_entities(Batmans)

# ------
# Basic insert entities:
#     We insert the `rings_entities` into our collection, into partition `American`, with ids we provide.
#     Here, we pass parameter '_async=True' to insert data asynchronously, and return a `Future` object which
#     has method `result()` to obtain result values and `done()` to wait util the invoked function(here is insert())
#     exit.
# ------

print("\n----------insert rings films----------")
insert_future = client.bulk_insert(collection_name, rings_entities, rings_ids, partition_tag="American", _async=True)
insert_future.result()


# ------
# Basic insert entities:
#     We insert the `batman_entities` into our collection, into partition `American`, with ids we provide.
#     Here, we pass parameter '_async=True' to insert data asynchronously, pass parameters '_callback=batman_insert_cb'
#     to provide callback function which could be called if data was inserted successfully.
#
#     The passing parameters of callback function are return values of insert().
# ------

def batman_insert_cb(inserted_ids):
    print("Films about Batman are inserted and the ids are: {}".format(inserted_ids))


insert_future = client.insert(collection_name, batman_entities, batman_ids, partition_tag="American", _async=True,
                              _callback=batman_insert_cb)
insert_future.done()

# ------
# Basic insert entities:
#     After insert entities into collection, we need to flush collection to make sure its on disk,
#     so that we are able to retrive it.
# ------

print("\n----------flush----------")
flush_future = client.flush([collection_name], _async=True)
flush_future.result()

# ------
# Basic hybrid search entities:
#      Getting films by id is not enough, we are going to get films based on vector similarities.
#      Let's say we have a film with its `embedding` and we want to find `top3` films that are most similar
#      with it by L2 distance.
#      Other than vector similarities, we also want to obtain films that:
#        `released year` term in 2002 or 2003,
#        `duration` larger than 250 minutes.
#
#      Milvus provides Query DSL(Domain Specific Language) to support structured data filtering in queries.
#      For now milvus suppots TermQuery and RangeQuery, they are structured as below.
#      For more information about the meaning and other options about "must" and "bool",
#      please refer to DSL chapter of our pymilvus documentation
#      (https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html).
# ------
query_embedding = [random.random() for _ in range(8)]
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
                    "embedding": {"topk": 1, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
}

# ------
# Basic hybrid search entities:
#     And we want to get all the fields back in reasults, so fields = ["duration", "release_year", "embedding"].
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
#
#      Here, we pass parameter '_async=True' to insert data asynchronously, and return a `Future` object.
# ------
print("\n----------search----------")
search_future = client.search(collection_name, dsl, _async=True)
search_results = search_future.result()


# ------
# Basic delete:
#     Now let's see how to delete things in Milvus.
#     You can simply delete entities by their ids.
#
#     After deleted, we invoke compact collection in a asynchronous way.
# ------
print("\n----------delete id = 1, id = 2----------")
client.delete_entity_by_id(collection_name, ids=[1, 4])
client.flush()  # flush is important
compact_future = client.compact(collection_name, _async=True)
compact_future.result()

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
