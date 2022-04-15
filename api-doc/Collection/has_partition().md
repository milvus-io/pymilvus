# has_partition()

This method verifies if a partition exists in the specified collection.

## Invocation

```python
has_partition(partition_name, timeout=None)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `partition_name`  | Name of the partition to verify                              | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |

## Return

A boolean value that indicates if the partition exists.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the specified partition does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.has_partition("novel")
```