# SearchFuture()

This is the constructor method to create a SearchFuture object.

## Invocation

```python
SearchFuture(future)
```

## Return

A SearchFuture object.

### Attributes

| Property             | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `cancel()`           | Cancel the search request                                    |
| `done()`             | Wait until search request done                               |
| `result(**kwargs)`   | Return the search result. It is a synchronous interface, and will wait until server respond or timeout occurs (if specified).                                     |

