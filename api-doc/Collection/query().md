# search()

This method conducts a vector query.

## Invocation

```python
query(expr, output_fields=None, partition_names=None, timeout=None, **kwargs)
```

## Parameters

| Parameter         | Description                                                   | Type               | Required |
| ----------------- | ------------------------------------------------------------- | ------------------ | -------- |
| `expr`            | Boolean expression to filter the data                         | String             | True     |
| `partition_names` | List of names of the partitions to search on. </br>All partition will be searched if it is left empty.                                                                              | list[String]       | False    |
| `output_fields`   | List of names of fields to output                             | list[String]       | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                                | Float              | False    |
| `kwargs` <ul><li>consistency_level</li><li>guarantee_timestamp</li><li>graceful_time</li><li>travel_timestamp</li></ul> | <br/><ul><li>Consistency level used in the search</li><li>Milvus searches on the data view before this timestamp when it is provided. Otherwise, it searches the most updated data view. It can be only used in Customized level of consistency.</li><li>PyMilvus will use current timestamp minus the graceful_time as the guarantee_timestamp for search. It can be only used in Bounded level of consistency.</li><li>Timestamp that is used for Time Travel. Users can specify a timestamp in a search to get results based on a data view at a specified point in time.</li></ul> | <br/><ul><li>String/Integer</li><li>Integer</li><li>Integer</li><li>Integer</li></ul>    | False    |

## Return

A list that contains all results.

## Raises

- `RpcError`: error if gRPC encounter an error.
- `ParamError`: error if the parameters are invalid.
- `DataTypeNotMatchException`: error if wrong type of data is passed to server.
- `BaseException`: error if the return result from server is not ok.

## Example

```python
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
res = collection.query(
  expr = "book_id in [2,4,6,8]", 
  output_fields = ["book_id", "book_intro"],
  consistency_level="Strong"
)
sorted_res = sorted(res, key=lambda k: k['book_id'])
sorted_res
```