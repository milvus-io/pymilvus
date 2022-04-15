# MutationFuture()

This is the constructor method to create a MutationFuture object.

## Invocation

```python
MutationFuture(future)
```

## Return

A MutationFuture object.

### Attributes

| Property             | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `cancel()`           | Cancel the search request                                    |
| `done()`             | Wait until search request done                               |
| `result(**kwargs)`   | Return the search result. It is a synchronous interface, and will wait until server respond or timeout occurs (if specified).                                     |

