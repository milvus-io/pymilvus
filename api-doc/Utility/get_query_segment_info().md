# get_query_segment_info()

This method checks the information of the segments in the query nodes.

## Invocation

```python
get_query_segment_info(collection_name, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection_name` | Name of the collection to check                              | String                          | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to check the segments                 | String                          | False    |

## Return

`QuerySegmentInfo`.
