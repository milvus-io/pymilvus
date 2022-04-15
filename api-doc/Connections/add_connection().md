# add_connection()

This method adds a Milvus connection.

## Invocation

```python
add_connection(**kwargs)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `kwargs` <ul><li>alias</li><li>host</li><li>port</li></ul>     | <br/><ul><li>Connection alias</li><li>IP address of the Milvus connection</li><li>Port of the Milvus connection</li></ul> | <br/><ul><li>String</li><li>String</li><li>Integer</li></ul>                  | <br/><ul><li>True</li><li>True</li><li>True</li></ul>    |

## Return

No return.

## Raises



## Example

```python
from pymilvus import connections
connections.add_connection(
  default={"host": "localhost", "port": "19530"}
)
```