# Intro to Indexes

For more detailed informations about indexes, please refer to [Milvus documentation index chapter.](https://milvus.io/docs/v0.11.0/index.md)

### Index building parameters and search parameters.
|Index           |building parameter|search parameter|
|:--------------:|:----------------:|:--------------:|
|FLAT            |NULL              | NULL           |
|BIN_FLAT        |NULL              | NULL           |
|IVF_FLAT        |nlist             | nprobe         |
|BIN_IVF_FLAT    |nlist             |nprobe          |
|IVF_PQ          |m, nlist          | nprobe         |
|IVF_SQ8         |nlist             | nprobe         |
|IVF_SQ8_HYBRID  |nlist             | nprobe         |
|ANNOY           |n_trees           | search_k       |
|HNSW            |M, efConstruction | ef             |
|RHNSW_PQ        |M, efConstruction,PQM | ef         |
|RHNSW_SQ        |M, efConstruction | ef             |
|RNSG            |search_length, out_degree, candidate_pool_size, knng|search_length|

---

**Index building  parameters with description, validation range.**

|Building parameter|description|validation range|
|:--------------:|:----------------:|:---:|
|nlist|Number of cluster units|[1, 65536]|
|m|Number of factors of product quantization|CPU-only Milvus: `m ≡ dim (mod m)`; GPU-enabled Milvus: `m` ∈ {1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96}, and (dim / m) ∈ {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32}. (`m` x 1024) ≥ `MaxSharedMemPerBlock` of your graphics card.|
|M|Maximum degree of the node|[4, 64]|
|efConstruction|Take effect in stage of index construction, search scope|[8, 512]|
|PQM|m for PQ|CPU-only Milvus: `m ≡ dim (mod m)`|
|search_length|Number of query iterations|[10, 300]|
|out_degree|Maximum out-degree of the node|[5, 300]|
|candidate_pool_size|Candidate pool size of the node|[50, 1000]|
|knng|Number of nearest neighbors|[5, 300]|
|n_trees|The number of methods of space division|[1, 1024]|

**Search in index parameters with description, validation range and recommender values.**

|Search parameter|description|validation range|
|:--------------:|:----------------:|:--------------:|
|nprobe|Number of inverted file cell to probe|CPU: [1, `nlist`], GPU: [1, min(2048, `nlist`)]|
|ef|                         Search scope                         |[`topk`, 32768]|
|search_length|Number of query iterations|[10, 300]|
|search_k|The number of nodes to search. -1 means 5% of the whole data|{-1} U [`top_k`, n*`n_trees`]|