# Milvus Python SDK

Using Milvus python sdk for Milvus

Download
---
```$
$ pip install pymilvus
```

## Import

```python
from milvus import Milvus, Prepare, IndexType
```

## Getting started

Initial a `Milvus` instance and  `connect` to the sever

```python
>>> milvus = Milvus()

>>> milvus.connect(host='SERVER-HOST', port='SERVER-PORT')
Status(code=0, message="Success")
```
Once successfully connected, you can get the version of server

```python
>>> milvus.server_version()
0.0.0  # this is example version, the real version may vary
```
---

Add a new `table`


First using `Prepare` to create param
```python
>>> param = Prepare.table_schema(table_name='test01', dimension=256, index_type=IndexType.IDMAP,
                                    store_raw_vector=False)
```
Then create `table`
```python
>>> milvus.create_table(param)
Status(message='Table test01 created!', code=0)
```

Describe the table we just created
```python
>>> milvus.describe_table('test01')
(Status(code=0, message='Success!'), TableSchema(table_name='test01',dimension=256, index_type=1, store_raw_vector=False))
```

---

Add vectors into table `test01`

First Prepare `binary vectors` of 256-dimension.

- Note that `random`, `struct` and `pprint` we used here is for creating fake vectors data and pretty print, you may not need them in your project

```python
>>> import random
>>> import struct
>>> from pprint import pprint

>>> dim = 256  # Dimension of the vector

# Initialize 20 vectors of 256-dimension
>>> fake_vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
>>> vectors = Prepare.records(fake_vectors)  # This will transfer fake_vector to binary data
```

Then add vectors into table `test01`
```python
>>> status, ids = milvus.add_vectors(table_name='test01', records=vectors)
>>> print(status)
Status(code=0, message='Success')
>>> pprint(ids) # List of ids returned
23455321135511233
12245748929023489
...
```
---
Search vectors

```python
# prepare 5 vectors of 256-dimension
>>> q_records = Prepare.records([random.random() for _ in range(dim)] for _ in range(5)]
```

Then get results
```python
>>> status, results = milvus.search_vectors(table_name='test01', query_records=q_records, top_k=10)
>>> print(status)
Status(code=0, message='Success')
>>> pprint(results) # Searched top_k vectors
```


Disconnect with the server
```python
>>> milvus.disconnect()
Status(code=0, message='Success')
```

---

## Example python
There is a small example in `examples/example.py`, you can find more guide there.

Build docs
```$
$ sphinx-build -b html doc/en/ doc/en/build
```


If you encounter any problems or bugs, please add new issues
