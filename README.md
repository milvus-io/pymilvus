
# Milvus Python SDK

[![version](https://img.shields.io/pypi/v/pymilvus.svg?color=blue)](https://pypi.org/project/pymilvus/)
[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/pymilvus/blob/master/LICENSE)

Python SDK for [Milvus](https://github.com/milvus-io/milvus). To contribute code to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) first.

For detailed SDK documentation, refer to [API Documentation](https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.2.6/index.html).


<!-- TOC -->

- [New features](#new-features)
- [Get started](#get-started)
    - [Prerequisites](#prerequisites)
    - [Install pymilvus](#install-pymilvus)
    - [Examples](#examples)
- [Basic operations](#basic-operations)
    - [Import modules](#import-modules)
    - [Connect to Milvus server](#connect-to-milvus-server)
    - [Create a table](#create-a-table)
    - [Get table information](#get-table-information)
    - [Insert vectors](#insert-vectors)
    - [Get the sum of vectors](#get-the-sum-of-vectors)
    - [Load vectors into memory](#load-vectors-into-memory)
    - [Create index](#create-index)
    - [Get index information](#get-index-information)
    - [Search vectors](#search-vectors)
    - [Create a partition](#create-a-partition)
    - [Insert vectors to a partition](#insert-vectors-to-a-partition)
    - [Get partitions of a table](#get-partitions-of-a-table)
    - [Search vectors in a partition](#search-vectors-in-a-partition)
    - [Drop index](#drop-index)
    - [Drop a table](#drop-a-table)
    - [Disconnect from Milvus server](#disconnect-from-milvus-server)

<!-- /TOC -->

## New features

* Add new metric type `HAMMING`, `JACCARD`, `TANIMOTO` for binary vectors. examples about binary vectors in `examples/example_binary.py`

## Get started

### Prerequisites

pymilvus only supports Python 3.5 or higher.

### Install pymilvus

You can install pymilvus via `pip` or `pip3` for Python3:

```shell
$ pip3 install pymilvus
```

The following table shows Milvus versions and recommended pymilvus versions:

|Milvus version| Recommended pymilvus version |
|:-----:|:-----:|
| 0.3.0 | 0.1.13|
| 0.3.1 | 0.1.25|
| 0.4.0 | 0.2.2 |
| 0.5.0 | 0.2.3 |
| 0.5.1 | 0.2.3 |
| 0.5.2 | 0.2.3 |
| 0.5.3 | 0.2.5 |
| 0.6.0 | 0.2.6 |


You can install a specific version of pymilvus by:

```shell
$ pip install pymilvus==0.2.7
```

You can upgrade `pymilvus` to the latest version by:

```shell
$ pip install --upgrade pymilvus
```

### Examples

Refer to [/examples](/examples) for more example programs.

## Basic operations

### Import modules

```python
from milvus import Milvus, IndexType, MetricType, Status
```

### Connect to Milvus server

```python
>>> milvus = Milvus()

>>> milvus.connect(host='localhost', port='19530')
Status(code=0, message='Successfully connected! localhost:19530')
```

Once successfully connected, you can get the version of Milvus server.

```python
>>> milvus.server_version()
(Status(code=0, message='Success'), '0.6.0')  # this is example version, the real version may vary
```


### Create a table

1. Set table parameters.

    ```python
    >>> dim = 32  # Dimension of the vector
    >>> param = {'table_name':'test01', 'dimension':dim, 'index_file_size':1024, 'metric_type':MetricType.L2}
    ```

2. Create a table.

    ```python
    >>> milvus.create_table(param)
    Status(code=0, message='Create table successfully!')
    ```

### Get table information

```python
>>> milvus.describe_table('test01')
(Status(code=0, message='Describe table successfully!'), TableSchema(table_name='test01', dimension=32, index_file_size=1024, metric_type=<MetricType: L2>))
```

### Insert vectors

1. Create 20 vectors of 256-dimension.

    > Note: `random` and `pprint` are used for demonstration purposes only.

    ```python
    >>> import random
    >>> from pprint import pprint

    # Initialize 20 vectors of 256-dimension
    >>> vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
    ```

2. Insert vectors to table `test01`.

    ```python
    >>> status, ids = milvus.insert(table_name='test01', records=vectors)
    >>> print(status)
    Status(code=0, message='Add vectors successfully!')
    >>> pprint(ids) # List of ids returned
    [1571123848227800000,
     1571123848227800001,
        ...........
     1571123848227800018,
     1571123848227800019]
    ```
    
    Alternatively, you can specify user-defined IDs for all vectors:

    ```python
    >>> vector_ids = [i for i in range(20)]
    >>> status, ids = milvus.insert(table_name='test01', records=vectors, ids=vector_ids)
    >>> pprint(ids)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ```

### Get the sum of vectors

```python
>>> milvus.count_table('test01')
(Status(code=0, message='Success!'), 20)
```

### Load vectors into memory

```python
>>> milvus.preload_table('test01')
Status(code=0, message='')
```

### Create index

```python
>>> index_param = {'index_type': IndexType.FLAT, 'nlist': 128}
>>> milvus.create_index('test01', index_param)
Status(code=0, message='Build index successfully!')
```

### Get index information

```python
>>> milvus.describe_index('test01')
(Status(code=0, message='Successfully'), IndexParam(_table_name='test01', _index_type=<IndexType: FLAT>, _nlist=128))
```

### Search vectors

```python
# create 5 vectors of 32-dimension
>>> q_records = [[random.random() for _ in range(dim)] for _ in range(5)]
>>> status, results = milvus.search(table_name='test01', query_records=q_records, top_k=1, nprobe=8)
>>> print(status)
Status(code=0, message='Search vectors successfully!')
>>> pprint(results) # Searched top_k vectors
[
[(id:15, distance:2.855304718017578),
 (id:16, distance:3.040700674057007)],
[(id:11, distance:3.673950433731079),
 (id:15, distance:4.183730602264404)],
      ........
[(id:6, distance:4.065953254699707),
 (id:1, distance:4.149323463439941)]
]
```
---

### Create a partition

Create a new partition named `partition01` under table `test01`, and specify tag `tag01`.

```python
>>> milvus.create_partition('test01', 'partition01', 'tag01')
Status(code=0, message='OK')
```

### Insert vectors to a partition

```python
>>> status = milvus.insert('demo01', vectors, partition_tag="tag01")
>>> status
(Status(code=0, message='Add vectors successfully!')
```

### Get partitions of a table

```python
milvus.show_partitions(table_name='test01')
```

### Search vectors in a partition

```python
>>> milvus.search(table_name='test01', query_records=q_records, top_k=1, nprobe=8, partition_tags=['tag01'])
```

If you do not specify `partition_tags`, Milvus searches the whole table.

### Drop index

```python
>>> milvus.drop_index('test01')
Status(code=0, message='')
```

### Drop a table

```python
>>> milvus.drop_table(table_name='test01')
Status(code=0, message='Delete table successfully!')
```

### Disconnect from Milvus server

```python
>>> milvus.disconnect()
Status(code=0, message='Disconnect successfully')
```
