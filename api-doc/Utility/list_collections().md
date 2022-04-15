# list_collections()

This method lists all collections in Milvus.

## Invocation

```python
list_collections(timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to check the collection               | String                          | False    |

## Return

A list of collection names.
