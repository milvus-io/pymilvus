# disconnect()

This method disconnects the client from the specified Milvus connection.

## Invocation

```python
disconnect(alias)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `alias`      | Alias of the connection to disconnect from                   | String                          | True     |

## Return

No return.

## Raises



## Example

```python
from pymilvus import connections
connections.disconnect("default")
```