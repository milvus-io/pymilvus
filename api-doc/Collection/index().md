# index()

This method gets the index object in the collection.

## Invocation

```python
index()
```


## Return

The index object.

## Raises

- `CollectionNotExistException`: error if the collection does not exist.
- `BaseException`: error if the specified partition does not exist.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.index()
```