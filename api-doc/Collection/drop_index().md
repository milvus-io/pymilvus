# drop_index()

This method drops the index and its corresponding index file in the collection.

## Invocation

```python
drop_index(timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |


## Return

No return.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the index does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.drop_index()
```