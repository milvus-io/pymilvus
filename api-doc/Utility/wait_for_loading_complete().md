# wait_for_loading_complete()

This method blocks all other operations until loading is done, exception is raised, or timeout is triggered.

## Invocation

```python
wait_for_loading_complete(collection_name, partition_names=None, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to load                               | String                          | True     |
| `partition_names` | Name of the partition(s) to load                             | list[String]                    | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to load the collection                | String                          | False    |

## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `PartitionNotExistException`: error if the partition does not exist.
