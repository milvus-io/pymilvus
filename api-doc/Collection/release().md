# release()

This method releases the specified collection from memory.

## Invocation

```python
release(timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |

## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the collection has not been loaded to memory.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.release()
```