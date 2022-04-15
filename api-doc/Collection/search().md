# search()

This method conducts a vector similarity search.

## Invocation

```python
search(data, anns_field, param, limit, expr=None, partition_names=None, output_fields=None, timeout=None, round_decimal=-1, **kwargs)
```

## Parameters

| Parameter         | Description                                                   | Type               | Required |
| ----------------- | ------------------------------------------------------------- | ------------------ | -------- |
| `data`            | Data to search with                                           | list[list[Float]]  | True     |
| `anns_field`      | Name of the vector field to search on                         | String             | True     |
| `param`           | Specific search parameter(s) of the index on the vector field | Dict               | True     |
| `limit`           | Number of nearest records to return                           | Integer            | True     |
| `expr`            | Boolean expression to filter the data                         | String             | False    |
| `partition_names` | List of names of the partitions to search on. </br>All partition will be searched if it is left empty.                         | list[String]            | False    |
| `output_fields`   | List of names of fields to output                             | list[String]       | False    |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                  | Float              | False    |
| `round_decimal`   | Number of the decimal places of the returned distance         | Integer            | False    |
| `kwargs` <ul><li>_async</li><li>_callback</li><li>consistency_level</li><li>guarantee_timestamp</li><li>graceful_time</li><li>travel_timestamp</li></ul> | <br/><ul><li>Boolean value to indicate if to invoke asynchronously</li><li>Function that will be invoked after server responds successfully. It takes effect only if `_async` is set to `True`.</li><li>Consistency level used in the search</li><li>Milvus searches on the data view before this timestamp when it is provided. Otherwise, it searches the most updated data view. It can be only used in Customized level of consistency.</li><li>PyMilvus will use current timestamp minus the graceful_time as the guarantee_timestamp for search. It can be only used in Bounded level of consistency.</li><li>Timestamp that is used for Time Travel. Users can specify a timestamp in a search to get results based on a data view at a specified point in time.</li></ul> | <br/><ul><li>Bool</li><li>Function</li><li>String/Integer</li><li>Integer</li><li>Integer</li><li>Integer</li></ul>    | False    |

## Return

A SearchResult object, an iterable, 2d-array-like class whose first dimension is the number of vectors to query (`nq`), and the second dimension is the number of limit (`topk`).

## Raises

- `RpcError`: error if gRPC encounter an error.
- `ParamError`: error if the parameters are invalid.
- `DataTypeNotMatchException`: error if wrong type of data is passed to server.
- `BaseException`: error if the return result from server is not ok.

## Example

```python
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
results = collection.search(
	data=[[0.1, 0.2]], 
	anns_field="book_intro", 
	param=search_params, 
	limit=10, 
	expr=None,
	consistency_level="Strong"
)
results[0].ids
results[0].distances
```