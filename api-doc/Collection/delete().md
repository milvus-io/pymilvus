# delete()

This method deletes entities from a specified collection.

## Invocation

```python
delete(expr, partition_name=None, timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `expr`            | Boolean expression that specifies the primary keys of the entities to delete | String          | True     |
| `partition_name`  | Name of the partition to delete data from                    | String                          | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                 | Float                           | False    |

## Return

A MutationResult object.

### Properties

| Property        | Description                                                  | Type                            |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| `delete_count`  | Number of the entities to delete                             | Integer                         |

## Raises

- `RpcError`: error if gRPC encounter an error.
- `ParamError`: error if the parameters are invalid.
- `BaseException`: error if the return result from server is not ok.

## Example

```python
expr = "book_id in [0,1]"
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.delete(expr)
```