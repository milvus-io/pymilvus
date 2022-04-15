# partition()

This method gets the specified partition object.

## Invocation

```python
partition(partition_name)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `partition_name`  | Name of the partition to get                                 | String                          | true     |


## Return

The specified partition object.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the specified partition does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.partition("novel")
```