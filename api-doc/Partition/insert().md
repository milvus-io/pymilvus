# insert()

This method inserts data into a specified partition.

## Invocation

```python
insert(data, timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `data`            | Data to insert                                               | list-like(list, tuple)          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |

## Return

A MutationResult object.

### Properties

| Property        | Description                                                  | Type                            |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| `insert_count`  | Number of the inserted entities                              | Integer                         |
| `primary_keys`  | List of the primary keys of the inserted entities            | list[String]                    |

## Raises

`PartitionNotExistException`: error if the partition does not exist.

## Example

```python
import random
data = [
  [i for i in range(2000)],
  [i for i in range(10000, 12000)],
  [[random.random() for _ in range(2)] for _ in range(2000)],
]
from pymilvus import Partition
partition = Partition("book", "novel", "")
partition.insert(data)
```