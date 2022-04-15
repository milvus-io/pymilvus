# calc_distance()

This method calculate distance between vectors.

## Invocation

```python
calc_distance(vectors_left, vectors_right, params=None, timeout=None, using='default')
```

## Parameters

| Parameter         | Description                                                  | Type                            | Required |
| ----------------- | ------------------------------------------------------------ | ------------------------------- | -------- |
| `vectors_left`    | Vectors on the left to the operator                          | Dict                            | True     |
| `vectors_right`   | Vectors on the right to the operator                         | Dict                            | True     |
| `params`          | Parameters used for calculation. Key-value pair parameters: Key: "metric_type"/"metric"; Value: "L2"/"IP"/"HAMMING"/"TANIMOTO", default is "L2". Key: "sqrt"; Value: `true` or `false`, default is false - only for "L2" distance. Key: "dim"; Value: Integer - set this value if dimension is not a multiple of 8, otherwise the dimension will be calculated by list length - only for "HAMMING" and "TANIMOTO".                                                                        | Dict                            | True     |
| `timeout`         | An optional duration of time in seconds to allow for the RPC. If it is set to None, the client keeps waiting until the server responds or error occurs.                                               | Float                           | False    |
| `using`           | Milvus Connection used to drop the collection                | String                          | False    |

### Vector example

```
{"ids": [1, 2, 3, .... n], "collection": "c_1", "partition": "p_1", "field": "v_1"}
```

```
{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]} or {"bin_vectors": [b'', b'N', ... b'Ê']}
```

### Params example

```
{"metric_type": "L2", "sqrt": true}
```

```
{"metric_type": "IP"}
```

```
{"metric_type": "HAMMING", "dim": 17}
```

```
{"metric_type": "TANIMOTO"}
```

## Return

A two-dimensional array indicates the distances.

## Example

```python
vectors_left = {
    "ids": [0, 1], 
    "collection": "book", 
    "partition": "_default", 
    "field": "book_intro"
}
import random
external_vectors = [[random.random() for _ in range(2)] for _ in range(4)]
vectors_right = {"float_vectors": external_vectors}
params = {
    "metric": "IP", 
    "dim": 2
}
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.load()
from pymilvus import utility
results = utility.calc_distance(
    vectors_left=vectors_left, 
    vectors_right=vectors_right, 
    params=params
)
print(results)
```
