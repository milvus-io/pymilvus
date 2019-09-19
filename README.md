
# Milvus Python SDK -- pymilvus

[![version](https://img.shields.io/pypi/v/pymilvus.svg?color=blue)](https://pypi.org/project/pymilvus/)
[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/pymilvus/blob/master/LICENSE)

Using Milvus python sdk for Milvus
Download
---
Pymilvus only supports `python >= 3.4`, is fully tested under 3.4, 3.5, 3.6, 3.7.


Pymilvus can be downloaded via `pip`. If no use, try `pip3`
```$
$ pip install pymilvus
```
Different versions of Milvus and lowest/highest pymilvus version supported accordingly

|Milvus version| Lowest pymilvus version supported|Highest pymivus version supported|
|:---:|:---:|:---:|
|0.3.0|-|0.1.13|
|0.3.1|0.1.14|0.1.25|
|0.4.0|0.2.0|-|

You can download a specific version by:
```$
$ pip install pymilvus==0.2.0
```

If you want to upgrade `pymilvus` to newest version
```$
$ pip install --upgrade pymilvus
```
## Import

```python
from milvus import Milvus, IndexType, MetricType, Status
```

## Getting started

Initial a `Milvus` instance and  `connect` to the sever

```python
>>> milvus = Milvus()

>>> milvus.connect(host='SERVER-HOST', port='SERVER-PORT')
Status(code=0, message='Successfully connected!')
```
Once successfully connected, you can get the version of server

```python
>>> milvus.server_version()
(Status(code=0, message='Success'), 0.4.0)  # this is example version, the real version may vary
```
---

Add a new `table`


First set param
```python
>>> param = {'table_name':'test01', 'dimension':256, 'index_file_size':1024, 'metric_type':MetricType.L2}
```
Then `create table`
```python
>>> milvus.create_table(param)
Status(code=0, message='Create table successfully!')
```

Describe the table we just created
```python
>>> milvus.describe_table('test01')
(Status(code=0, message='Describe table successfully!'), TableSchema(table_name='test01', dimension=256, index_file_size=1024, metric_type=<MetricType: L2>))
```

---

Add vectors into table `test01`

First create 20 vectors of 256-dimension.

- Note that `random` and `pprint` we used here is for creating fake vectors data and pretty print, you may not need them in your project

```python
>>> import random
>>> from pprint import pprint

>>> dim = 256  # Dimension of the vector

# Initialize 20 vectors of 256-dimension
>>> vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
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
You can also specify vectors id
```python
>>> vector_ids = [i for i in range(20)]
>>> status, ids = milvus.add_vectors(table_name='test01', records=vectors, ids=vector_ids)
>>> pprint(ids)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```
Get vectors num
```python
>>> milvus.get_table_row_count('test01')
(Status(code=0, message='Success!'), 20)
```
---
Load vectors into memory
```python
>>> milvus.preload_table('test01')
Status(code=0, message='')
```
---
Create index
```python
>>> index_param = {'index_type': IndexType.IVFLAT, 'nlist': 16384}
>>> milvus.create_index('test01', index_param)
Status(code=0, message='Build index successfully!')
```
Then show index information
```python
>>> client.describe_index('test01')
(Status(code=0, message='Successfully'), IndexParam(_table_name='test01', _index_type=<IndexType: IVFLAT>, _nlist=16384))
```
---
Search vectors

```python
# create 5 vectors of 256-dimension
>>> q_records = [[random.random() for _ in range(dim)] for _ in range(5)]
```

Then get results
```python
>>> status, results = milvus.search_vectors(table_name='test01', query_records=q_records, top_k=1, nprobe=16)
>>> print(status)
Status(code=0, message='Search vectors successfully!')
>>> pprint(results) # Searched top_k vectors
[
[QueryResult(id=0, distance=34.85963439941406)],
[QueryResult(id=0, distance=36.73900604248047)],
[QueryResult(id=0, distance=34.35655975341797)],
[QueryResult(id=18, distance=36.19701385498047)],
[QueryResult(id=5, distance=39.11549758911133)]
]
```
---
Drop index
```python
>>> milvus.drop_index('test01')
Status(code=0, message='')
```
---
Delete vectors by date range
>>> 
```python
>>> milvus.delete_vectors_by_range('test01', '2019-06-01', '2020-01-01')
Status(code=0, message='')
```
---
Delete the table we just created

```python
>>> milvus.delete_table(table_name='test01')
Status(code=0, message='Success')
```
Disconnect with the server
```python
>>> milvus.disconnect()
Status(code=0, message='Success')
```

---

## Example python
There are some small examples in `examples/`, you can find more guide there.

Build docs
```$
$ sphinx-build -b html doc/en/ doc/en/build
```


If you encounter any problems or bugs, please open new issues
