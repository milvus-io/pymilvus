# has_collection()

This method checks if a collection exists.

## Invocation

```python
has_collection(collection_name, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to check                              | String                          | True     |
| `using`           | Milvus Connection used to check the collection               | String                          | False    |

## Return

Boolean value that indicates if the collection exists.
