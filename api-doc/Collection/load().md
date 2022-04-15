# load()

This method loads the specified collection to memory (for search or query).

## Invocation

```python
load(partition_names=None, timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `partition_names` | Name of the partition(s) to load                             | list[String]                    | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                 | Float                           | False    |
| `kwargs` <ul><li>_async</li></ul> | <br/><ul><li>Boolean value to indicate if to invoke asynchronously</li></ul> | Bool | False    |

## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `ParamError`: error if the parameters are invalid.
- `BaseException`: error if the specified partition does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.load()
```