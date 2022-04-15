# wait_for_index_building_complete()

This method blocks all other operations until index building is done, exception is raised, or timeout is triggered.

## Invocation

```python
wait_for_index_building_complete(collection_name, index_name='', timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to load                               | String                          | True     |
| `index_name`      | Name of the index to build. Default index will be checked if it is left blank. | String        | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to load the collection                | String                          | False    |

## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `IndexNotExistException`: error if the index does not exist.