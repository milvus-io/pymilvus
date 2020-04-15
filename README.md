
# Milvus Python SDK

[![version](https://img.shields.io/pypi/v/pymilvus.svg?color=blue)](https://pypi.org/project/pymilvus/)
[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/pymilvus/blob/master/LICENSE)

Python SDK for [Milvus](https://github.com/milvus-io/milvus). To contribute code to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) first.

For detailed SDK documentation, refer to [API Documentation](https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.2.10/index.html).


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
- [Insert/Delete vectors in collections/partitions](#insertdelete-vectors-in-collectionspartitions)
    - [Insert vectors in a collection](#insert-vectors-in-a-collection)
    - [Insert vectors in a partition](#insert-vectors-in-a-partition)
    - [Delete vectors by ID](#delete-vectors-by-id)
- [Flush data in one or multiple collections to disk](#flush-data-in-one-or-multiple-collections-to-disk)
- [Compact all segments in a collection](#compact-all-segments-in-a-collection)
- [Search vectors in collections/partitions](#search-vectors-in-collectionspartitions)
    - [Search vectors in a collection](#search-vectors-in-a-collection)
    - [Search vectors in a partition](#search-vectors-in-a-partition)
- [Disconnect from the Milvus server](#disconnect-from-the-milvus-server)

<!-- /TOC -->

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


You can install a specific version of pymilvus by:

```shell
$ pip install pymilvus==0.2.10
```

You can upgrade `pymilvus` to the latest version by:

```shell
$ pip install --upgrade pymilvus
```

### Examples

Refer to [examples](/examples) for more example programs.

## Basic operations

## Connect to the Milvus server

1. Import pymilvus.

   ```python
   # Import pymilvus
   >>> from milvus import Milvus, IndexType, MetricType, Status
   ```

2. Connect to Milvus server by using one of the following methods:

   ```python
   # Connect to Milvus server
   >>> client = Milvus(host='localhost', port='19530')
   >>> client.connect()
   ```

   > Note: In the above code, default values are used for `host` and `port` parameters. Feel free to change them to the IP address and port you set for Milvus server.
   
   ```python
   >>> client = Milvus(uri='tcp://localhost:19530')
   ```

## Create/Drop collections

### Create a collection

1. Prepare collection parameters.

   ```python
   # Prepare collection parameters
   >>> param = {'collection_name':'test01', 'dimension':128, 'index_file_size':1024, 'metric_type':MetricType.L2}
   ```

2. Create collection `test01` with dimension size as 256, size of the data file for Milvus to automatically create indexes as 1024, and metric type as Euclidean distance (L2).

   ```python
   # Create a collection
   >>> client.create_collection(param)
   ```

### Drop a collection

```python
# Drop collection
>>> client.drop_collection(collection_name='test01')
```

## Create/Drop partitions in a collection

### Create a partition

You can split collections into partitions by partition tags for improved search performance. Each partition is also a collection.

```python
# Create partition
>>> client.create_partition(collection_name='test01', partition_tag='tag01')
```

Use `show_partitions()` to verify whether the partition is created.

```python
# Show partitions
>>> client.show_partitions(collection_name='test01')
```

### Drop a partition

```python
>>> client.drop_partition(collection_name='test01', partition_tag='tag01')
```

## Create/Drop indexes in a collection

### Create an index

> Note: In production, it is recommended to create indexes before inserting vectors into the collection. Index is automatically built when vectors are being imported. However, you need to create the same index again after the vector insertion process is completed because some data files may not meet the `index_file_size` and index will not be automatically built for these data files.

1. Prepare index parameters. The following command uses `IVF_FLAT` index type as an example.

   ```python
   # Prepare index param
   >>> ivf_param = {'nlist': 16384}
   ```

2. Create an index for the collection.

   ```python
   # Create index
   >>> client.create_index('test01', IndexType.IVF_FLAT, ivf_param)
   ```

### Drop an index

```python
>>> client.drop_index('test01')
```

## Insert/Delete vectors in collections/partitions

### Insert vectors in a collection

1. Generate 20 vectors of 256 dimension.

   ```python
   >>> import random
   >>> dim = 128
   # Generate 20 vectors of 256 dimension
   >>> vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
   ```

2. Insert the list of vectors. If you do not specify vector ids, Milvus automatically generates IDs for the vectors.

   ```python
   # Insert vectors
   >>> client.insert(collection_name='test01', records=vectors)
   ```

   Alternatively, you can also provide user-defined vector ids:

   ```python
   >>> vector_ids = [id for id in range(20)]
   >>> client.insert(collection_name='test01', records=vectors, ids=vector_ids)
   ```

### Insert vectors in a partition

```python
>>> client.insert('test01', vectors, partition_tag="tag01")
```

To verify the vectors you have inserted, use `get_vector_by_id()`. Assume you have a vector with the following ID.

```python
>>> status, vector = client.get_vector_by_id(collection_name='test01', vector_id=0)
```

### Delete vectors by ID

Assume you have some vectors with the following IDs:

```python
>>> ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

You can delete these vectors by:

```python
>>> client.delete_by_id('test01', ids)
```

## Flush data in one or multiple collections to disk

When performing operations related to data changes, you can flush the data from memory to disk to avoid possible data loss. Milvus also supports automatic flushing, which runs at a fixed interval to flush the data in all collections to disk. You can use the [Milvus server configuration file](https://milvus.io/docs/reference/milvus_config.md) to set the interval.

```python
>>> client.flush(['test01'])
```

## Compact all segments in a collection

A segment is a data file that Milvus automatically creates by merging inserted vector data. A collection can contain multiple segments. If some vectors are deleted from a segment, the space taken by the deleted vectors cannot be released automatically. You can compact segments in a collection to release space.

```python
>>> client.compact(collection_name='test01')
```

## Search vectors in collections/partitions

### Search vectors in a collection

1. Prepare search parameters.

```python
>>> search_param = {'nprobe': 16}
```

2. Search vectors.

```python
# create 5 vectors of 32-dimension
>>> q_records = [[random.random() for _ in range(dim)] for _ in range(5)]
# search vectors
>>> client.search(collection_name='test01', query_records=q_records, top_k=2, params=search_param)
```

### Search vectors in a partition

```python
# create 5 vectors of 32-dimension
>>> q_records = [[random.random() for _ in range(dim)] for _ in range(5)]
>>> client.search(collection_name='test01', query_records=q_records, top_k=1, partition_tags=['tag01'], params=search_param)
```

> Note: If you do not specify `partition_tags`, Milvus searches the whole collection.

## Disconnect from the Milvus server

```python
>>> client.disconnect()
```
