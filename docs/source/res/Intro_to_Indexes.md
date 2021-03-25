
[Milvus](https://github.com/milvus-io) support to create index to accelerate vector approximate search. 

To learn how to create an index by python client, see method [create_index()](api.html#milvus.Milvus.create_index) and 
[index example](https://github.com/milvus-io/pymilvus/tree/1.0/examples) .

For more detailed information about indexes, please refer to [Milvus documentation index chapter.](https://milvus.io/docs/index.md)

To learn how to choose an appropriate index for your application scenarios, please read [How to Select an Index in Milvus](https://medium.com/@milvusio/how-to-choose-an-index-in-milvus-4f3d15259212).

To learn how to choose an appropriate index for a metric, see [Distance Metrics](https://www.milvus.io/docs/metric.md).

## Vector Index

- `FLAT`_
- `IVF_FLAT`_
- `IVF_SQ8`_
- `IVF_SQ8_H`_
- `IVF_PQ`_
- `HNSW`_
- `ANNOY`_
- `RNSG`_


### FLAT
If FLAT index is used, the vectors are stored in an array of float/binary data without any compression. during 
searching vectors, all indexed vectors are decoded sequentially and compared to the query vectors.

FLAT index provides 100% query recall rate. Compared to other indexes, it is the most efficient indexing method 
when the number of queries is small.

The inserted and index-inbuilt vectors and index-dropped vectors are regard as built with ``FLAT``.

- building parameters: 
  **N/A**
  
```python
# FLAT
client.create_index(collection_name, IndexType.FLAT)
```

- search parameters: 
  **N/A**
  
```python
# FLAT
client.search(collection_name,
              1,
              query_vectors
)
```

### IVF_FLAT

**IVF** (*Inverted File*) is an index type based on quantization. It divides the points in space into `nlist`
units by clustering method. During searching vectors, it compares the distances between the target vector
and the center of all the units, and then select the `nprobe` nearest unit. Then, it compares all the vectors
in these selected cells to get the final result.

IVF_FLAT is the most basic IVF index, and the encoded data stored in each unit is consistent with the original data.

- building parameters:

  **nlist**: Number of cluster units.

```python
# IVF_FLAT
client.create_index(collection_name, IndexType.IVF_FLAT, {
    "nlist": 100      # int. 1~65536
})
```

- search parameters:

  **nprobe**: Number of inverted file cell to probe.

```python
# IVF_FLAT
client.search(collection_name,
              1,
              query_vectors, 
              params={
                "nprobe": 8 # int. 1~nlist(cpu), 1~min[2048, nlist](gpu)
              }
)
```

### IVF_PQ

**PQ** (*Product Quantization*) uniformly decomposes the original high-dimensional vector space into
Cartesian products of `m` low-dimensional vector spaces, and then quantizes the decomposed low-dimensional
vector spaces. In the end, each vector is stored in `m` × `nbits` bits. Instead of calculating the distances 
between the target vector and the center of all the units, product quantization enables the calculation of 
distances between the target vector, and the clustering center of each low-dimensional space and greatly reduces
the time complexity and space complexity of the algorithm.

IVF_PQ performs IVF index clustering, and then quantizes the product of vectors. Its index file is even
smaller than IVF_SQ8, but it also causes a loss of accuracy during searching.

- building parameters:

  **nlist**: Number of cluster units.

  **m**: Number of factors of product quantization. **CPU-only** Milvus: `m ≡ dim (mod m)`; **GPU-enabled** Milvus: `m` ∈ {1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96}, and (dim / m) ∈ {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32}. (`m` x 1024) ≥ `MaxSharedMemPerBlock` of your graphics card.

  **nbits**: Number of bits in which each low-dimensional vector is stored.

```python
# IVF_PQ
client.create_index(collection_name, 
                    IndexType.IVF_PQ, 
                    {
                      "nlist": 100,     # int. 1~65536
                      "m": 8            # int. 1~16. 8 by default
                    }
)
```

- search parameters:

  **nprobe**: Number of inverted file cell to probe.

```python
# IVF_PQ
client.search(collection_name,
              1, 
              query_vectors, 
              params={
                "nprobe": 8 # int. 1~nlist(cpu), 1~min[2048, nlist](gpu)
              }
)

```

### IVF_SQ8

**IVF_SQ8** does scalar quantization for each vector placed in the unit based on IVF. Scalar quantization
converts each dimension of the original vector from a 4-byte floating-point number to a 1-byte unsigned integer,
so the IVF_SQ8 index file occupies much less space than the IVF_FLAT index file.
However, scalar quantization results in a loss of accuracy during searching vectors.

- building parameters:
  
  **nlist**: Number of cluster units.

```python
# IVF_SQ8
client.create_index(collection_name, 
                    IndexType.IVF_SQ8, 
                    {
                      "nlist": 100      # int. 1~65536
                    }
)
```

- search parameters:

  **nprobe**: Number of inverted file cell to probe.

```python
# IVF_SQ8
client.search(collection_name,
              1, 
              query_vectors, 
              params={
                "nprobe": 8       # int. 1~nlist(cpu), 1~min[2048, nlist](gpu)
              }
)

```

### IVF_SQ8_H

Optimized version of IVF_SQ8 that requires both CPU and GPU to work. Unlike IVF_SQ8, IVF_SQ8_H uses a GPU-based
coarse quantizer, which greatly reduces time to quantize.

IVF_SQ8H is an IVF_SQ8 index that optimizes query execution.

The query method is as follows:

- If `nq` ≥ `gpu_search_threshold`, GPU handles the entire query task.
- If `nq` < `gpu_search_threshold`, GPU handles the task of retrieving the `nprobe` nearest unit in the IVF
index file, and CPU handles the rest.

- building parameters:

  **nlist**: Number of cluster units.

```python
# IVF_SQ8_H
client.create_index(collection_name, 
                    IndexType.IVF_SQ8_H, 
                    {
                      "nlist": 100      # int. 1~65536
                    }
)
```

- search parameters:

  **nprobe**: Number of inverted file cell to probe.

```python
# IVF_SQ8_H
client.search(collection_name,
              1, 
              query_vectors, 
              params={
                "nprobe": 8       # int. 1~nlist(cpu), 1~min[2048, nlist](gpu)
              }
)
```

### ANNOY

**ANNOY** (*Approximate Nearest Neighbors Oh Yeah*) is an index that uses a hyperplane to divide a
high-dimensional space into multiple subspaces, and then stores them in a tree structure.

When searching for vectors, ANNOY follows the tree structure to find subspaces closer to the target vector,
and then compares all the vectors in these subspaces (The number of vectors being compared should not be
less than `search_k`) to obtain the final result. Obviously, when the target vector is close to the edge of
a certain subspace, sometimes it is necessary to greatly increase the number of searched subspaces to obtain
a high recall rate. Therefore, ANNOY uses `n_trees` different methods to divide the whole space, and searches
all the dividing methods simultaneously to reduce the probability that the target vector is always at the edge of the subspace.

- building parameters:

  **n_trees**:  The number of methods of space division.

```python
# ANNOY
client.create_index(collection_name, 
                    IndexType.ANNOY, 
                    {
                      "n_trees": 8      # int. 1~1024
                    }
)
```

- search parameters:

  **search_k**: The number of nodes to search. -1 means 5% of the whole data.

```python
# ANNOY
client.search(collection_name,
              1,
              query_vectors, 
              params={
                "search_k": -1    # int. {-1} U [top_k, n*n_trees], n represents vectors count.
              }
)
```

### HNSW

**HNSW** (*Hierarchical Navigable Small World Graph*) is a graph-based indexing algorithm. It builds a
multi-layer navigation structure for an image according to certain rules. In this structure, the upper
layers are more sparse and the distances between nodes are farther; the lower layers are denser and 
he distances between nodes are closer. The search starts from the uppermost layer, finds the node closest
to the target in this layer, and then enters the next layer to begin another search. After multiple iterations,
it can quickly approach the target position.

In order to improve performance, HNSW limits the maximum degree of nodes on each layer of the graph to `M`.
In addition, you can use `efConstruction` (when building index) or `ef` (when searching targets) to specify a search range.

- building parameters:

  **M**: Maximum degree of the node.

  **efConstruction**: Take the effect in stage of index construction.

```python
# HNSW
client.create_index(collection_name, 
                    IndexType.HNSW, 
                    {
                      "M": 16,              # int. 4~64
                      "efConstruction": 40  # int. 8~512
                    }
)
```

- search parameters:

  **ef**: Take the effect in stage of search scope, should be larger than `top_k`.

```python
# HNSW
client.search(collection_name,
              1, 
              query_vectors, 
              params={
                "ef": 64          # int. top_k~32768
              }
)
```


### RNSG

**RNSG** (*Refined Navigating Spreading-out Graph*) is a graph-based indexing algorithm. It sets the center
position of the whole image as a navigation point, and then uses a specific edge selection strategy to control
the out-degree of each point (less than or equal to `out_degree`). Therefore, it can reduce memory usage and
quickly locate the target position nearby during searching vectors.

The graph construction process of NSG is as follows:

1. Find `knng` nearest neighbors for each point.
2. Iterate at least `search_length` times based on `knng` nearest neighbor nodes to select `candidate_pool_size` possible nearest neighbor nodes.
3. Construct the out-edge of each point in the selected `candidate_pool_size` nodes according to the edge selection strategy.

The query process is similar to the graph building process. It starts from the navigation point and iterates at least `search_length` times to get the final result.

- building parameters:

  **search_length**: Number of query iterations.

  **out_degree**: Maximum out-degree of the node.

  **candidate_pool_size**: Candidate pool size of the node.

  **knng**: Number of nearest neighbors

```python
# RNSG
client.create_index(collection_name, 
                    IndexType.RNSG, 
                    {
                      "search_length": 60,         # int. 10~300
                      "out_degree": 30,            # int. 5~300
                      "candidate_pool_size": 300,  # int. 50~1000
                      "knng": 50                   # int. 5~300
                    }
)
```

- search parameters:

	**search_length**: Number of query iterations

```python
# RNSG
client.search(collection_name,
              1, 
              query_vectors, 
              params={
                "search_length": 100  # int. 10~300
              }
)
```
