# create_partition()

This method creates a partition with the specified name.

## Invocation

```python
create_partition(partition_name, description="")
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `partition_name`  | Name of the partition to create                              | String                          | True     |
| `description`     | Description of the partition to create                       | String                          | False    |

## Return

The newly created partition object.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the specified partition does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.create_partition("novel")
```