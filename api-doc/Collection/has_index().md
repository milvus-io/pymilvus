# has_index()

This method verifies if a specified index exists.

## Invocation

```python
has_index(timeout=None)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |

## Return

A boolean value that indicates if the index exists.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.has_index()
```