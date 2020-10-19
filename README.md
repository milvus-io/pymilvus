
# Milvus Python SDK

[![version](https://img.shields.io/pypi/v/pymilvus.svg?color=blue)](https://pypi.org/project/pymilvus/)
[![Downloads](https://pepy.tech/badge/pymilvus)](https://pepy.tech/project/pymilvus)
[![Downloads](https://pepy.tech/badge/pymilvus/month)](https://pepy.tech/project/pymilvus/month)
[![Downloads](https://pepy.tech/badge/pymilvus/week)](https://pepy.tech/project/pymilvus/week)
[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/pymilvus/blob/master/LICENSE)

Python SDK for [Milvus](https://github.com/milvus-io/milvus). To contribute code to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) first.

For detailed SDK documentation, refer to [API Documentation](https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html).


<!-- TOC -->

- [New features](#new-features)
- [Get started](#get-started)
    - [Prerequisites](#prerequisites)
    - [Install pymilvus](#install-pymilvus)
    - [Examples](#examples)
- [Basic operations](#basic-operations)
    - [Connect to the Milvus server](#connect-to-the-milvus-server)
- [Create/Drop collections](#createdrop-collections)
    - [Create a collection](#create-a-collection)
    - [Drop a collection](#drop-a-collection)
- [Create/Drop partitions in a collection](#createdrop-partitions-in-a-collection)
    - [Create a partition](#create-a-partition)
    - [Drop a partition](#drop-a-partition)
- [Create/Drop indexes in a collection](#createdrop-indexes-in-a-collection)
    - [Create an index](#create-an-index)
    - [Drop an index](#drop-an-index)
- [Insert/Delete vectors in collections/partitions](#insertdelete-entities-in-collectionspartitions)
    - [Insert entities in a collection](#insert-entities-in-a-collection)
    - [Insert entities in a partition](#insert-entities-in-a-partition)
    - [Delete entities by ID](#delete-entities-by-id)
- [Flush data in one or multiple collections to disk](#flush-data-in-one-or-multiple-collections-to-disk)
- [Compact all segments in a collection](#compact-all-segments-in-a-collection)
- [Search entities in collections/partitions](#search-entities-in-collectionspartitions)
    - [Search entities in a collection](#search-entities-in-a-collection)
    - [Search entities in a partition](#search-entities-in-a-partition)
- [Disconnect from the Milvus server](#close-client)
- [FAQ](#faq)

<!-- /TOC -->

## New features

- remove `get_index_info`, the index info can be obtained by `get_collection_info`

- add `get_collection_stats`, more detailed collection stats.

## Get started

### Prerequisites

pymilvus only supports Python 3.5 or higher.

### Install pymilvus

You can install pymilvus via `pip` or `pip3` for Python3:

```shell
$ pip3 install pymilvus
```

The following collection shows Milvus versions and recommended pymilvus versions:

|Milvus version| Recommended pymilvus version |
|:-----:|:-----:|
| 0.3.0 | 0.1.13|
| 0.3.1 | 0.1.25|
| 0.4.0 | 0.2.2 |
| 0.5.0 | 0.2.3 |
| 0.5.1 | 0.2.3 |
| 0.5.2 | 0.2.3 |
| 0.5.3 | 0.2.5 |
| 0.6.0 | 0.2.6, 0.2.7 |
| 0.7.0 | 0.2.8 |
| 0.7.1 | 0.2.9 |
| 0.8.0 | 0.2.10 |
| 0.9.0 | 0.2.11 |
| 0.9.1 | 0.2.12 |
| 0.10.0 | 0.2.13 |
| >=0.10.1, <0.11.0 | 0.2.14 |
| 0.11.0 | 0.3.0 |


You can install a specific version of pymilvus by:

```shell
$ pip install pymilvus==0.3.0
```

You can upgrade `pymilvus` to the latest version by:

```shell
$ pip install --upgrade pymilvus
```

### Examples

Refer to [examples](/examples) for more example programs.

## Basic operations

### Connect to the Milvus server

1. Import pymilvus.

```python
# Import pymilvus
>>> from milvus import Milvus, DataType
```

2. Create a client to Milvus server using one of the following methods:

```python
# Connect to Milvus server
>>> client = Milvus(host='localhost', port='19530')
```

   > Note: In the above code, default values are used for `host` and `port` parameters.
Feel free to change them to the IP address and port you set for Milvus server.

```python
>>> client = Milvus(uri='tcp://localhost:19530')
```

## Create/Drop collections

### Create a collection

1. Prepare collection parameters.

```python
# create collection name
>>> collection_name = 'test01'

# create a collection of 4 fields, fields A, B and C are int type fields
# and Vec is a float vector field.
# segment_row_limit is default as 524288 if not specified
>>> collection_param = {
...    "fields": [
...        {"name": "A", "type": DataType.INT32},
...        {"name": "B", "type": DataType.INT32},
...        {"name": "C", "type": DataType.INT64},
...        {"name": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}
...    ],
...    "segment_row_limit": 4096,
...    "auto_id": True
... }
```

2. Create collection `test01` with dimension of 128, size of the data file for Milvus to automatically
create indexes as 4096. If `metric_type` isn't offered, default metric type is Euclidean distance (L2).
For `FLOAT_VECTOR` field, `dim` is a must.

```python
# Create a collection
>>> client.create_collection(collection_name, collection_param)
```

3. You can check collection info by `get_collection_info`
```python
>>> info = client.get_collection_info('test01')
>>> info
{'fields': [
    {'name': 'A', 'type': <DataType.INT32: 4>, 'params': {}, 'indexes': [{}]},
    {'name': 'C', 'type': <DataType.INT64: 5>, 'params': {}, 'indexes': [{}]},
    {'name': 'B', 'type': <DataType.INT32: 4>, 'params': {}, 'indexes': [{}]},
    {'name': 'Vec', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 128},
     'indexes': [{}]}
    ],
 'auto_id': True,
 'segment_row_limit': 4096
}
```

You can see from the info, there is an `auto_id` option in collection info, and its `True` by default.
So if you have your own ids and don't want auto generated ids, you may want to set `auto_id` to `False`
while creating collections.



### Drop a collection

```python
# Drop collection
>>> status = client.drop_collection('test01')
>>> status
Status(code=0, message='OK')
```

## Create/Drop partitions in a collection

### Create a partition

You can split collections into partitions by partition tags for improved search performance.

```python
# Create partition
>>> client.create_partition(collection_name='test01', partition_tag='tag01')
```

Use `list_partitions()` to verify whether the partition is created.

```python
# Show partitions
>>> partitions = client.list_partitions(collection_name='test01')
>>> partitions
['_default', 'tag01']
```

### Drop a partition

```python
# Drop partitions
>>> status = client.drop_partition(collection_name='test01', partition_tag='tag01')
Status(code=0, message='OK')
```


## Create/Drop indexes in a collection

### Create an index

> Note: In production, it is recommended to create indexes before inserting vectors into the collection.
Index is automatically built when vectors are being imported. However, you need to create the same index
again after the vector insertion process is completed because some data files may not meet the
`index_file_size` and index will not be automatically built for these data files.

1. Create an index of `IVF_FLAT` with `nlist = 100` for the collection.

```python
# Create index
>>> status = client.create_index('test01', "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})
>>> status
Status(code=0, message='OK')   
```

2. You can check index info by `get_collection_info`

```python
>>> info = client.get_collection_info('test01')
>>> info
{'fields': [
    {'name': 'A', 'type': <DataType.INT32: 4>, 'params': {}, 'indexes': [{}]},
    {'name': 'C', 'type': <DataType.INT64: 5>, 'params': {}, 'indexes': [{}]},
    {'name': 'B', 'type': <DataType.INT32: 4>, 'params': {}, 'indexes': [{}]},
    {'name': 'Vec',
        'type': <DataType.FLOAT_VECTOR: 101>,
        'params': {'dim': 128, 'metric_type': 'L2'},
        'indexes': [{'index_type': 'IVF_FLAT', 'metric_type': 'L2', 'params': {'nlist': 100}}]}],
 'auto_id': True,
 'segment_row_limit': 4096
}
```

### Drop an index

```python
# Drop an index of a specific field "Vec"
>>> status = client.drop_index('test01', "Vec")
Status(code=0, message='OK')
```

## Insert/Delete entities in collections/partitions

### Insert entities in a collection

1. Generate 5000 vectors of 128 dimension and an integer list.

```python
>>> import random
>>> num = 5000

# Generate a list of integer.
>>> list_of_int = [random.randint(0, 255) for _ in range(num)]
# Generate 20 vectors of 128 dimension
>>> vectors = [[random.random() for _ in range(128)] for _ in range(num)]
```

2. Create hybrid entities

```python
>>> hybrid_entities = [
   {"name": "A", "values": list_of_int, "type": DataType.INT32},
   {"name": "B", "values": list_of_int, "type": DataType.INT32},
   {"name": "C", "values": list_of_int, "type": DataType.INT64},
   {"name": "Vec", "values": vectors, "type":DataType.FLOAT_VECTOR}
]
```

3. Insert the hybrid entities.

If you create a new collection with `auto_id = True`, Milvus automatically generates IDs for the vectors.

```python
# Insert vectors
>>> ids = client.insert('test01', hybrid_entities)
```

If you create a new collection with `auto_id = False`, you have to provide user-defined vector ids:

```python
# Generate fake custom ids
>>> vector_ids = [id for id in range(num)]
# Insert to the non-auto-id collection
>>> ids = client.insert('test01', hybrid_entities, ids=vector_ids)
```

**The examples below are based on collection with `auto_id = True`.**

### Insert entities in a partition

```python
>>> inserted_vector_ids = client.insert('test01', hybrid_entities, partition_tag="tag01")
```

To verify the entities you have inserted, use `get_entity_by_id()`. 

```python
>>> entities = client.get_entity_by_id(collection_name='test01', ids=inserted_vector_ids[:10])
```

### Delete entities by ID

You can delete these entities by:

```python
>>> status = client.delete_entity_by_id('test01', ids[:10])
>>> status
Status(code=0, message='OK')
```

## Flush data in one or multiple collections to disk

When performing operations related to data changes, you can flush the data from memory to disk to avoid possible data loss. Milvus also supports automatic flushing, which runs at a fixed interval to flush the data in all collections to disk. You can use the [Milvus server configuration file](https://milvus.io/docs/reference/milvus_config.md) to set the interval.

```python
>>> client.flush(['test01'])
```

## Compact all segments in a collection

A segment is a data file that Milvus automatically creates by merging inserted vector data. A collection can contain multiple segments. If some vectors are deleted from a segment, the space taken by the deleted vectors cannot be released automatically. You can compact segments in a collection to release space.

```python
>>> status = client.compact('test01')
>>> status
Status(code=0, message='OK')
```

## Search entities in collections/partitions

### Search entities in a collection

1. Prepare search parameters. `"term"` and `"range"` is optional, `"params"` in `"vector"` stands for index params.


```python
# This dsl will search topk `entities` that are
# close to vectors[:1] searched by `IVF_FLAT` index with `nprobe = 10` and `metric_type = L2`,
# AND field "A" in [1, 2, 5],
# AND field "B" greater than 1 less than 100
>>> dsl = {
...     "bool": {
...         "must":[
...             {
...                 "term": {"A": [1, 2, 5]}
...             },
...             {
...                 "range": {"B": {"GT": 1, "LT": 100}}
...             },
...             {
...                 "vector": {
...                    "Vec": {"topk": 10, "query": vectors[:1], "metric_type": "L2", "params": {"nprobe": 10}}
...                 }
...             }
...         ]
...     }
... }

```
A search without hybrid conditions with `IVF_FLAT` index would be like:

```python
>>> dsl = {
...     "bool": {
...         "must":[
...             {
...                 "vector": {
...                    "Vec": {"topk": 10, "query": vectors[:1], "metric_type": "L2", "params": {"nprobe": 10}}
...                 }
...             }
...         ]
...     }
... }

```

A `FLAT` search doesn't need index params, so the query would be like:

```python
>>> dsl = {
...     "bool": {
...         "must":[
...             {
...                 "vector": {
...                    "Vec": {"topk": 10, "query": vectors[0], "metric_type": "L2"}
...                 }
...             }
...         ]
...     }
... }

```


2. Search entities.

With `fields=["B"]`, not only can you get entity ids and distances, but also values of a spacific field B.

```python
# search entities and get entity field B back
>>> results = client.search('test01', dsl, fields=["B"])
```

You can obtain ids, distances and fields by entities in results.

```python
# Results consist of number-of-query entities
>>> entities = results[0]

# Entities consists of topk entity
>>> entity = entities[0]

# You can get all ids and distances by entities
>>> all_ids = entities.ids
>>> all_distances = entities.distances

# Or you can get them one by one by entity
>>> a_id = entity.id
>>> a_distance = entity.distance
>>> a_field = entity.entity.B # getattr(entity.entity, "B")
```

> Note: If you don't provide fields in search, you will only get ids and distances.

### Search entities in a partition

```python
# Search entities in a partition `tag01`
>>> client.search(collection_name='test01', dsl=dsl, partition_tags=['tag01'])
```

> Note: If you do not specify `partition_tags`, Milvus searches the whole collection.

## Close client

```python
>>> client.close()
```

## FAQ

> I'm getting random "socket operation on non-socket" errors from gRPC when connecting to Milvus from an application served on Gunicorn

Make sure to set the environment variable `GRPC_ENABLE_FORK_SUPPORT=1`. For reference, see https://zhuanlan.zhihu.com/p/136619485
