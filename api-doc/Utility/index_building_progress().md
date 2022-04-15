# index_building_progress()

This method shows the index building progress.

## Invocation

```python
index_building_progress(collection_name, index_name='', using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to build index                        | String                          | True     |
| `index_name`      | Name of the index to build. Default index will be checked if it is left blank. | String        | False    |
| `using`           | Milvus Connection used to build the index                    | String                          | False    |

## Return

A dict type contains the number of the indexed entities and the total entity number. 

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `IndexNotExistException`: error if the index does not exist.

