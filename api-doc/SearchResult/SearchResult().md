# SearchResult()

This is the constructor method to create a SearchResult.

## Invocation

```python
SearchResult(query_result=None)
```

## Return

A SearchResult object.

### Attributes

| Property             | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| `iter(self)`         | Iterate the search result. Each iteration returns a Hits corresponding to a query. |
| `self[item]`         | Return the Hits corresponding to the nth query                                     |
| `len(self)`          | Return the `nq` of search result                                                   |

