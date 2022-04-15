# delete()

This method deletes entities from a specified partition.

## Invocation

```python
delete(expr, timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `expr`            | Boolean expression that specifies the primary keys of the entities to delete | String          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                 | Float                           | False    |

## Return

A MutationResult object.

### Properties

| Property        | Description                                                  | Type                            |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| `delete_count`  | Number of the entities to delete                             | Integer                         |

## Raises

- `RpcError`: error if gRPC encounter an error.
- `ParamError`: error if the parameters are invalid.
- `BaseException`: error if the return result from server is not ok.

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
expr = "book_id in [0,1]"
partition.delete(expr)
```