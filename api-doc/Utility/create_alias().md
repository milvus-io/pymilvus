# create_alias()

This method specifies an alias for a collection. Alias cannot be duplicated. Same alias cannot be assigned to different collections. Instead, you can specify multiple aliases for each collection.

## Invocation

```python
create_alias(collection_name, alias, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to create alias                       | String                          | True     |
| `alias`           | Alias to create                                              | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to check the segments                 | String                          | False    |


## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if failed to create the alias.