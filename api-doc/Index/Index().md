# Index()

This is the constructor method to build an index on the specified field with the specified index parameters.

## Invocation

```python
Index(collection, field_name, index_params, **kwargs)
```

## Parameters

| Parameter      | Description                                                  | Type                            | Required |
| -------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection`   | Name of the collection                                       | String                          | True     |
| `field_name`   | Name of the field to build index on                          | String                          | True     |
| `index_params` | Milvus Connection used to create the collection              | Dict                            | True     |


## Return

A new index object created with the specified parameters.

### Properties

| Property          | Description                                                  | Type                            |
| ----------------- | ------------------------------------------------------------ | ------------------------------- |
| `params`          | Index parameters                                             | Dict                            |
| `collection_name` | Name of the collection                                       | String                          |
| `field_name`      | Name of the indexed field                                    | String                          |


## Example

```python
from pymilvus import Index
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
index = Index("book", "book_intro", index_params)
```