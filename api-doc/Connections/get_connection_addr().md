# get_connection_addr()

This method retrieves the configuration of the specified Milvus connection.

## Invocation

```python
get_connection_addr(alias)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `alias`      | Alias of the connection to retrieve                          | String                          | True     |

## Return

The configuration of the specified Milvus connection

## Raises



## Example

```python
from pymilvus import connections
connections.get_connection_addr("default")
```