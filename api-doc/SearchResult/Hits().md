# Hits()

This is the constructor method to create a Hits object.

## Invocation

```python
Hits(hits)
```

## Return

A Hits object.

### Attributes

| Property             | Description                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `iter(self)`         | Iterate the Hits object. Each iteration returns a Hit which represent a record corresponding to the query. |
| `self[item]`         | Return the kth Hit corresponding to the query                                                              |
| `len(self)`          | Return the number of hit record                                                                            |
| `ids`                | Return the primary keys of all search results                                                              |
| `distances`          | Return the distances of all hit record                                                                     |
