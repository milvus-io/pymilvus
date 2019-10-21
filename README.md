
# Milvus Python SDK -- pymilvus

[![version](https://img.shields.io/pypi/v/pymilvus.svg?color=blue)](https://pypi.org/project/pymilvus/)
[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/pymilvus/blob/master/LICENSE)

Using Milvus python sdk for Milvus
Download
---
Pymilvus only supports `python >= 3.5`, is fully tested under 3.5, 3.6, 3.7.


Pymilvus can be downloaded via `pip`. If no use, try `pip3`
```$
$ pip install pymilvus
```
Different versions of Milvus and lowest/highest pymilvus version supported accordingly

|Milvus version| Lowest pymilvus version supported|Highest pymivus version supported|
|:-----:|:-----:|:-----:|
| 0.3.0 |   -   | 0.1.13|
| 0.3.1 | 0.1.14| 0.1.25|
| 0.4.0 | 0.2.0 | 0.2.2 |
| 0.5.0 | 0.2.3 |   -   |

You can download a specific version by:
```$
$ pip install pymilvus==0.2.3
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

>>> milvus.connect(host='localhost', port='19530')
Status(code=0, message='Successfully connected! localhost:19530')
```
Once successfully connected, you can get the version of server

```python
>>> milvus.server_version()
(Status(code=0, message='Success'), '0.5.0')  # this is example version, the real version may vary
```
---

Add a new `table`


First set param
```python
>>> dim = 32  # Dimension of the vector
>>> param = {'table_name':'test01', 'dimension':dim, 'index_file_size':1024, 'metric_type':MetricType.L2}
```
Then `create table`
```python
>>> milvus.create_table(param)
Status(code=0, message='Create table successfully!')
```

Describe the table we just created
```python
>>> milvus.describe_table('test01')
(Status(code=0, message='Describe table successfully!'), TableSchema(table_name='test01', dimension=32, index_file_size=1024, metric_type=<MetricType: L2>))
```

---

Add vectors into table `test01`

First create 20 vectors of 256-dimension.

- Note that `random` and `pprint` we used here is for creating fake vectors data and pretty print, you may not need them in your project

```python
>>> import random
>>> from pprint import pprint

# Initialize 20 vectors of 256-dimension
>>> vectors = [[random.random() for _ in range(dim)] for _ in range(20)]
```

Then add vectors into table `test01`
```python
>>> status, ids = milvus.add_vectors(table_name='test01', records=vectors)
>>> print(status)
Status(code=0, message='Add vectors successfully!')
>>> pprint(ids) # List of ids returned
[1571123848227800000,
 1571123848227800001,
    ...........
 1571123848227800018,
 1571123848227800019]
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
>>> index_param = {'index_type': IndexType.FLAT, 'nlist': 128}
>>> milvus.create_index('test01', index_param)
Status(code=0, message='Build index successfully!')
```
Then show index information
```python
>>> milvus.describe_index('test01')
(Status(code=0, message='Successfully'), IndexParam(_table_name='test01', _index_type=<IndexType: FLAT>, _nlist=128))
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
Drop index
```python
>>> milvus.drop_index('test01')
Status(code=0, message='')
```
---
Delete the table we just created

```python
>>> milvus.delete_table(table_name='test01')
Status(code=0, message='Delete table successfully!')
```
Disconnect with the server
```python
>>> milvus.disconnect()
Status(code=0, message='Disconnect successfully')
```

---

## Example python
There are some small examples in `examples/`, you can find more guide there.

Build docs
```$
$ sphinx-build -b html doc/en/ doc/en/build
```


If you encounter any problems or bugs, please open new issues
