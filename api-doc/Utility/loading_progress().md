# loading_progress()

This method shows the loading progress of sealed segments (in percentage).

## Invocation

```python
loading_progress(collection_name, partition_names=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to load                               | String                          | True     |
| `partition_names` | Name of the partition(s) to load                             | list[String]                    | False    |
| `using`           | Milvus Connection used to load the collection                | String                          | False    |

## Return

The loading progress (in percentage).

## Raises

`PartitionNotExistException`: error if the partition does not exist.

