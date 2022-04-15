# remove_connection()

This method removes the specified Milvus connection.

## Invocation

```python
remove_connection(alias)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `alias`      | Alias of the connection to remove                            | String                          | True     |

## Return

No return.

## Raises



## Example

```python
from pymilvus import connections
connections.remove_connection("default")
```