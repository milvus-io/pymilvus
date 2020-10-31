## Prerequisites
Before we start, there are some prerequisites.

Make sure that:
- You have a running Milvus instance.
- PyMilvus is correctly installed.

In Python shell, your installation of PyMilvus is OK if the following command doesn't raise an exception:
```python
>>> from milvus import Milvus, DataType
```

## Connect to Milvus
First of all, we need to make connection with Milvus server after importing.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

```python
>>> host = '127.0.0.1'
>>> port = '19530'
>>> client = Milvus(host, port)
```

After connecting, we are able to communicate with Milvus in the following ways. If you are confused about the
terminology, see [Milvus Terminology](https://milvus.io/docs/v0.11.0/terms.md) for explanations.

## Collection
Now let's create a new collection. Before we start, we can list all the collections already exist. For a brand
new Milvus running instance, the result should be empty.

```python
>>> client.list_collections()
[]
```
To create collection, we need to provide a unique collection name and some other parameters.
`collection_name` should be a unique string to collections already exist. `collection_param` consists of 3 components, 
they are `"fields"`, `"segment_row_limit"` and `"auto_id"`.

```python
>>> collection_name = 'demo_film_tutorial'
>>> collection_param = {
...     "fields": [
...         {"name": "release_year", "type": DataType.INT32},
...         {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
...     ],
...     "segment_row_limit": 4096,
...     "auto_id": False
... }
```
In the `collection_param`, there are 2 fields in`"fields"`, the name of the first field is "release_year", and
the type of the first field is `DataType.INT32`, it's a field to store release year of a film.

The name of the second field is "embedding", the type of it is `DataType.FLOAT_VECTOR`. It also has an extra
parameter "params" with dimension 8. It's a float vector field to store embedding of a film, and for a
FLOAT_VECTOR, **the "dim" in "params" is a must**.
You are also able to add "params" in other types of field as in FLOAT_VECTOR field.

Milvus controls the size of data segment according to the `"segment_row_limit"`, you can refer
[Storage Concepts](https://milvus.io/docs/v0.11.0/storage_concept.md) for more information about segment and
`segment_row_limit`, or you can refer to [create index](https://milvus.io/docs/v0.11.0/create_drop_index_python.md)
to see how `segment_row_limint` influence the index building.

`"auto_id"` is used to tell Milvus if we want ids auto-generated or ids user provided for each entity.
If `False`, it means we'll have to provide our own ids for entities while inserting, on the contrary,
if `True`, Milvus will generate ids automatically and we can't provide our own ids for entities.

## Create Collection

Now we are able to create a collection
```python
>>> client.create_collection(collection_name, collection_param)
```

Then you can list collections and 'demo_film_tutorial' will be in the result.
```python
>>> client.list_collections()
['demo_film_tutorial']
```

You can also get info of the collection.
> For a better output format, we use pprint to print the result.
 <div></div>

```python
>>> from pprint import pprint
>>> info = client.get_collection_info(collection_name)
>>> pprint(info)
{'auto_id': False,
 'fields': [{'indexes': [{}],
             'name': 'release_year',
             'params': {},
             'type': <DataType.INT32: 4>},
            {'indexes': [{}],
             'name': 'embedding',
             'params': {'dim': 8},
             'type': <DataType.FLOAT_VECTOR: 101>}],
 'segment_row_limit': 4096}

```
You can see from the output, all the infos are the same as we provide, but there's one more called `"indexes"`
we don't know yet.

This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
[example_index.py](https://github.com/milvus-io/pymilvus/blob/master/examples/example_index.py).

If you're already known about indexes from `example_index.py`, and you want a full lists of params supported
by PyMilvus, you check out [Index params](https://pymilvus.readthedocs.io/en/latest/param.html)
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our official website for
[Vector Index](https://milvus.io/docs/v0.11.0/index.md)

## Create Partition

If you don't create a partition, there will be a default one called "_default",  all the entities will be inserted
into the "_default" partition. You can check it by `list_partitions()`

```python
>>> client.list_partitions(collection_name)
['_default']
```

You can provide a partition tag to create your own partition.
```python
>>> client.create_partition(collection_name, "American")
>>> client.list_partitions(collection_name)
['American', '_default']
```
## Entities

An entities is a group of fields that correspond to real world objects. Here is an example of 3 entities structured
in list of dictionary.

```python
>>> import random
>>> The_Lord_of_the_Rings = [
...     {
...         "id": 1,
...         "title": "The_Fellowship_of_the_Ring",
...         "release_year": 2001,
...         "embedding": [random.random() for _ in range(8)]
...     },
...     {
...         "id": 2,
...         "title": "The_Two_Towers",
...         "release_year": 2002,
...         "embedding": [random.random() for _ in range(8)]
...     },
...     {
...         "id": 3,
...         "title": "The_Return_of_the_King",
...         "release_year": 2003,
...         "embedding": [random.random() for _ in range(8)]
...     }
... ]
```
## Insert Entities
To insert entities into Milvus, we need to group data from the same field like below.
```python
>>> ids = [k.get("id") for k in The_Lord_of_the_Rings]
>>> release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
>>> embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]
```

Then we are able to create hybrid entities to insert into Milvus.
```python
>>> hybrid_entities = [
...     # Milvus doesn't support string type yet, so we cannot insert "title".
...     {"name": "release_year", "values": release_years, "type": DataType.INT32},
...     {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
... ]
```

If the hybrid entities inserted successfully, `ids` we provided will be returned.
>If we create collection with `auto_id = True`, we can't provide ids of our own, and the returned `ids` is automatically generated by Milvus. If `partition_tag` isn't provided, these entities will be inserted into the "_default" partition.
 <div></div>

```python
>>> client.insert(collection_name, hybrid_entities, ids, partition_tag="American")
[1, 2, 3]
```
## Flush
After successfully inserting 3 entities into Milvus, we can `Flush` data from memory to disk so that we are
able to retrieve them. Milvus also performs an automatic flush with a fixed interval(1 second),
see [Data Flushing](https://milvus.io/docs/v0.11.0/flush_python.md).

You can flush multiple collections at one time, so be aware the parameter is a list.
```python
>>> client.flush([collection_name])
```

## Get Detailed information
After insert, we can get the detail of collection statistics info by `get_collection_stats`
> Again, we are using pprint to provide a better format.
 <div></div>

```python
info = client.get_collection_stats(collection_name)
pprint(info)
```
And result should be like:
```python
{'data_size': 18156,
 'partition_count': 2,
 'partitions': [{'data_size': 0,
                 'id': 13,
                 'row_count': 0,
                 'segment_count': 0,
                 'segments': None,
                 'tag': '_default'},
                {'data_size': 18156,
                 'id': 14,
                 'row_count': 3,
                 'segment_count': 1,
                 'segments': [{'data_size': 18156,
                               'files': [{'data_size': 4124,
                                          'field': '_id',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_49'},
                                         {'data_size': 5724,
                                          'field': '_id',
                                          'name': '_blf',
                                          'path': '/C_7/P_14/S_7/F_53'},
                                         {'data_size': 4112,
                                          'field': 'release_year',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_51'},
                                         {'data_size': 4196,
                                          'field': 'embedding',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_50'}],
                               'id': 7,
                               'row_count': 3}],
                 'tag': 'American'}],
 'row_count': 3}
```
## Count Entities

We can also count how many entities are there in the collection.
```python
>>> client.count_entities(collection_name)
3
```

## Get Entities
There are three ways to get entities from a collection.

### Get Entities by ID
You can get entities by their ids.
```python
films = client.get_entity_by_id(collection_name, ids=[1, 200])
```
If id exists, an entity will be returned. If id doesn't exist, `None` will be return. For the example above,
collection `"demo_film_tutorial"` has an entity(id = 1), but doesn't have an entity(id = 200), so the result
`films` will only have one entity, the other is `None`. You can get the entity fields like below.

```python
>>> for film in films:
...     if film is not None:
...         print(film.id, film.release_year, film.embedding)
```
The result may look like below. Because embeddings are random generated, so the value of embedding may differ.
```python
1 2001 [0.5146051645278931, 0.9257888197898865, 0.8659316301345825, 0.8082002401351929, 0.33681046962738037, 0.7135953307151794, 0.14593836665153503, 0.9224222302436829]
```

### Search Entities by Vector Similarity
You can get entities by vector similarity. Assuming we have a film like below, and we want to get top 2 films
that are most similar with it.

```python
>>> film_A = {
...     "title": "random_title",
...     "release_year": 2002,
...     "embedding": [random.random() for _ in range(8)]
... }
```

We need to prepare query DSL(Domain Specific Language) for this search, for more information about dos and don't
for Query DSL , please refer to PyMilvus documentation
[Query DSL](https://pymilvus.readthedocs.io/en/latest/query.html) chapter.

```python
>>> query = {
...     "bool": {
...         "must": [
...             {
...                 "vector": {
...                     "embedding": {"topk": 2, "query": [film_A.get("embedding")], "metric_type": "L2"}
...                 }
...             }
...         ]
...     }
... }
```

Then we are able to search by this query.
> If we don't provide anything in `"fields"`, there will only be ids and distances in the results. Only what we have provided in the `"fields"` can be obtained finally.
 <div></div>

```python
>>> results = client.search(collection_name, query, fields=["release_year"])
```

The returned `results` is a 1 * 2 structure, 1 for 1 entity querying, 2 for top 2. For more clarity, we obtain
the film as below. If you want to know how to deal with search result in a better way, you can refer to
[search result](https://pymilvus.readthedocs.io/en/latest/results.html) in PyMilvus doc.

```python
>>> entities = results[0]
>>> film_1 = entities[0]
>>> film_2 = entities[1]
```

Then how do we get ids, distances and fields? It's as below.

> Because embeddings are randomly generated, so the retrieved film id, distance and field may differ.
 <div></div>

```python
# id
>>> film_1.id
3
# distance
>>> film_1.distance
0.3749755918979645

# fields
>>> film_1.entity.release_year
2003
```

### Search Entities filtered by fields.

Milvus can also search entities back by vector similarity combined with fields filtering. Again we will be
using query DSL, please refer to PyMilvus documentation
[Query DSL](https://pymilvus.readthedocs.io/en/latest/query.html) for more information.

For the same `film_A`, now we also want to search back top2 most similar films, but with one more condition: 
the release year of films searched back should be the same as `film_A`. Here is how we organize query DSL.

```python
>>> query_hybrid = {
...     "bool": {
...         "must": [
...             {
...                 "term": {"release_year": [film_A.get("release_year")]}
...             },
...             {
...                 "vector": {
...                     "embedding": {"topk": 2, "query": [film_A.get("embedding")], "metric_type": "L2"}
...                 }
...             }
...         ]
...     }
... }
```

Then we'll do search as above. This time we will only get 1 film back because there is only 1 film whose release
year is 2002, the same as `film_A`. You can confirm it by `len()`.

```python
>>> results = client.search(collection_name, query_hybrid, fields=["release_year"])
>>> len(results[0])
1
```

We are also able to search back entities with fields in specific range like below, we want to get top 2 films
that are most similar with `film_A`, and we want the `release_year` of entities returned must be larger than
`film_A`'s `release_year`

```python
>>> query_hybrid = {
...     "bool": {
...         "must": [
...             {
...                 # "GT" for greater than
...                 "range": {"release_year": {"GT": film_A.get("release_year")}}
...             },
...             {
...                 "vector": {
...                     "embedding": {"topk": 2, "query": [film_A.get("embedding")], "metric_type": "L2"}
...                 }
...             }
...         ]
...     }
... }
```

This query will only get 1 film back too, because there is only 1 film whose release year is larger than 2002.

Again, for more information about query DSL, please refer to our documentation 
[Query DSL](https://pymilvus.readthedocs.io/en/latest/query.html).

## Deletion
Finally, let's move on to deletion in Milvus. We can delete entities by ID, we can drop a whole partition
(entities in the partition will be deleted as well), or we can drop the entire collection.

### Delete Entities by ID
You are able to delete entities by their ids.
```python
>>> client.delete_entity_by_id(collection_name, ids=[1, 2])
Status(code=0, message='OK')

>>> client.count_entities(collection_name)
1
```
### Drop a Partition

You can also drop a partition.
> Caution: once you drop a partition, all the data in this partition will be deleted too.
 <div></div>
  

```python
>>> client.drop_partition(collection_name, "American")
Status(code=0, message='OK')

>>> client.count_entities(collection_name)
0
```
### Drop a Collection
Finally, you can drop an entire collection.
> Caution: once you drop a collection, all the data in this collection will be deleted too.
 <div></div>

```python
>>> client.drop_collection(collection_name)
Status(code=0, message='OK')
```
