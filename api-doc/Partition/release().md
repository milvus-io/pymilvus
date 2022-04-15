# release()

This method releases the specified partition from memory.

## Invocation

```python
release(timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |

## Return

No return.

## Raises

`PartitionNotExistException`: error if the partition does not exist.

## Example

```python
from pymilvus import Partition
partition = Partition("book", "novel", "")
partition.load()
partition.release()
```