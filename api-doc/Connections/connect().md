# connect()

This method connects the client to a Milvus connection.

## Invocation

```python
connect(alias, **kwargs)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `alias`      | Connection alias                                             | String                          | True     |
| `kwargs` <ul><li>host</li><li>port</li></ul>     | <br/><ul><li>IP address of the Milvus connection</li><li>Port of the Milvus connection</li></ul> | <br/><ul><li>String</li><li>Integer</li></ul>                  | <br/><ul><li>True</li><li>True</li></ul>    |

## Return

No return.

## Raises

- `NotImplementedError` -- error if the handler in connection parameters is not gRPC.
- `ParamError` -- error if the parameter is invalid.
- `Exception` -- error if the server specified in parameters is not ready.

## Example

```python
from pymilvus import connections
connections.connect(
  alias="default", 
  host='localhost', 
  port='19530'
)
```