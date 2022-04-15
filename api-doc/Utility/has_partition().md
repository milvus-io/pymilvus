# has_partition()

This method checks if a partition exists in a specified collection.

## Invocation

```python
has_partition(collection_name, partition_name, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to check the partition                | String                          | True     |
| `partition_name`  | Name of the partition to check                               | String                          | True     |
| `using`           | Milvus Connection used to check the collection               | String                          | False    |

## Return

Boolean value that indicates if the partition exists.
