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

Initial a `milvus` instance and  `connect` to the sever

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

You can also add a new `table`


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

There is a small example in examples/example.py, you can find more guide there.

Build docs
```$
$ sphinx-build -b html doc/en/ doc/en/build
```

---

If you encounter any problems or bugs, please add new issues



