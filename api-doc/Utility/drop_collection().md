# has_collection()

This method drops a collection and the data within.

## Invocation

```python
drop_collection(collection_name, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to drop                               | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to drop the collection                | String                          | False    |

## Return

No return.

## Example

```python
from pymilvus import utility
utility.drop_collection("book")
```
