# Milvus Python SDK

Using Milvus python sdk for Milvus

## Build

### Requirements

Pymilvus only supports `python >= 3.4`, is fully tested under 3.4, 3.5, 3.6.

Python 3.7 can work, but is not fully tested yet.

### Install

Use `pip` or `pip3` to install pimilvus:

```$
$ pip install pymilvus
```

### Upgrade to newest version

```$
$ pip install --upgrade pymilvus
```

## Import

```python
from milvus import Milvus, IndexType, Status
```

## Getting started

Initial a `Milvus` instance and `connect` to the sever

```python
>>> milvus = Milvus()

>>> milvus.connect(host='SERVER-HOST', port='SERVER-PORT')
Status(code=0, message="Success")
```
Once successfully connected, you can get the version of server

```python
>>> milvus.server_version()
(Status(code=0, message='Success'), 0.3.0)  # this is example version, the real version may vary
```

### Add a `table`


1. Set parameters.
```python
>>> param = {'table_name'='test01', 'dimension'=256, 'index_type'=IndexType.FLAT, 'store_raw_vector'=False}
```
2. Create a `table`.
```python
>>> milvus.create_table(param)
Status(message='Table test01 created!', code=0)
```

3. Confirm table information.
```python
>>> milvus.describe_table('test01')
(Status(code=0, message='Success!'), TableSchema(table_name='test01',dimension=256, index_type=1, store_raw_vector=False))
```


### Load vectors into table `test01`

In case you don't have any available vectors, you can try creating 20 vectors of 256-dimension.

> Note: `random` and `pprint` is used for creating fake vectors data and pretty print. You may not need them in your project

```python
>>> import random
>>> from pprint import pprint

>>> dim = 256  # Dimension of the vector

# Initialize 20 vectors of 256-dimension
>>> fake_vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
```

Load vectors into table `test01`:
```python
>>> status, ids = milvus.add_vectors(table_name='test01', records=vectors)
>>> print(status)
Status(code=0, message='Success')
>>> pprint(ids) # List of ids returned
23455321135511233
12245748929023489
...
```

### Search vectors
1. Specify the search range. 
```python
# create 5 vectors of 256-dimension
>>> q_records = [[random.random() for _ in range(dim)] for _ in range(5)]
```

2. Search vectors and you can get such results:
```python
>>> status, results = milvus.search_vectors(table_name='test01', query_records=q_records, top_k=10)
>>> print(status)
Status(code=0, message='Success')
>>> pprint(results) # Searched top_k vectors
```


### Delete a table

```python
>>> milvus.delete_table(table_name='test01')
Status(code=0, message='Success')
```
Disconnect with the server
```python
>>> milvus.disconnect()
Status(code=0, message='Success')
```

## Example python
There are some small examples in `examples/`, you can find more guide there.

Build docs
```$
$ sphinx-build -b html doc/en/ doc/en/build
```


If you encounter any problems or bugs, please add new issues
