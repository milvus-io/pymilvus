# drop_alias()

This method alters an alias of a collection to another.

## Invocation

```python
drop_alias(alias, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `alias`           | Alias to drop                                                | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to check the segments                 | String                          | False    |


## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the alias does not exist.