# create_alias()

This method alters an alias of a collection to another.

## Invocation

```python
alter_alias(collection_name, alias, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to alter alias to                     | String                          | True     |
| `alias`           | Alias to alter                                               | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to check the segments                 | String                          | False    |


## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if failed to alter the alias.