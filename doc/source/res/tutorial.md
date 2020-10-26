## Prerequisites
Before we start, there are some prerequisites. Make sure that:
- You have a running Milvus instance.
- PyMilvus is correctly installed.

In Python shell, your installation of PyMilvus is ok if the following command doesn't raise an exception:
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
terminology, see [Milvus Terminology](https://milvus.io/docs/v0.11.0/terms.md) for explainations.

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

The name of the second field is "embedding", the type of it is `DataType.FLOAT_VECTOR`. It also has an extro
parameter "params" with dimention 8. It's a float vector field to store embedding of a film, and for a
FLOAT_VECTOR, **the "dim" in "params" is a must**.
You are also able to add "params" in other types of field as in FLOAT_VECTOR field.

Milvus controls the size of data segment according to the `"segment_row_limit"`, you can refer
[Storage Concepts](https://milvus.io/docs/v0.11.0/storage_concept.md) for more information about segment and
`segment_row_limit`, or you can refer to [create index](https://milvus.io/docs/v0.11.0/create_drop_index_python.md)
to see how `segment_row_limint` influence the index building.

`"auto_id"` is used to tell Milvus if we want ids auto-generated or ids user provided for each entity.
If `False`, it means we'll have to provide our own ids for entities while inserting, on the contrary,
if `True`, Milvus will generate ids automatically and we can't provide our own ids for entites.
 
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
>For a better output format, we use pprint to print the result.

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
You can see from the ouput, all the infos are the same as we provide, but there's one more called `"indexes"`
we don't know yet.

This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
[example_index.py](https://github.com/milvus-io/pymilvus/blob/master/examples/example_index.py).

If you're already known about indexes from `example_index.py`, and you want a full lists of params supported
by PyMilvus, you check out [Index params](https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/param.html)
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our offical website for
[Vector Index](https://milvus.io/docs/v0.11.0/index.md)

## Create Partition

If you don't create a partition, there will be a default one called "_default",  all the entites wil be inserted
into the "_default" partition. You can check it by `list_partitions()`

```python
>>> client.list_partitions(collection_name)
['_default']
```

You can provide a partition tag to create your own parition.
```python
>>> client.create_partition(collection_name, "American")
>>> client.list_partitions(collection_name)
['American', '_default']
```
## Entities

An entities is a group of fields that correspond to real world objects. Here is a example of 3 entities structured
by dictionary.

```python

```
## Insert Entities
## Flush
## Count Entites
## Get Entities
  ### Get Entities by ID
  ### Search Entities by Vector Similiarity
  ### Get Entities filtered by fields.
## Deletion
  ### Delete Entities by ID
  ### Compact
  ### Drop a Partition
  ### Drop a Collection
