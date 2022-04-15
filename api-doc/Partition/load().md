# load()

This method loads the specified partition to memory (for search or query).

## Invocation

```python
load(timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                 | Float                           | False    |
| `kwargs` <ul><li>_async</li></ul> | <br/><ul><li>Boolean value to indicate if to invoke asynchronously</li></ul> | Bool | False    |

## Return

No return.

## Raises

`InvalidArgumentException`: error if the argument is invalid.

## Example

```python
from pymilvus import Partition
partition = Partition("book", "novel", "")
partition.load()
```