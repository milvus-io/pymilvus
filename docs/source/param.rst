pymilvus index params
=====================


Module Contents
---------------
This module briefly illustrate parameters of each type of index. You can find detail example in `index examples <https://github.com/milvus-io/pymilvus/blob/master/examples/indexes>`_.


Index
------


FLAT/IVF_FLAT
^^^^^^^^^^^^^

**Create index param**

* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1, nlist]
  * recommended value: 32


IVF_PQ
^^^^^^^

**Create index param**

* m

  * m is decided by dim and have a couple of results. each result represent a kind of compress ratio.
  * Range:  [96, 64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 8, 4, 3, 2, 1]
  * recommended value: 12

* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1, nlist]
  * recommended value: 32


IVF_SQ8/IVF_SQ8H
^^^^^^^^^^^^^^^^^

**Create index param**


* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1, nlist]
  * recommended value: 32


HNSW
^^^^^

**Create index param**


* M

  * number of neighbor in index graph
  * Range: [5, 48]
  * recommended value: 16


* efConstruction

  * take effect in stage of index construction. The larger the value is, the more time cost during creatind index and the more accurate search results are.
  * Range: [100, 500]
  * recommended value: 500

**Search param**

* ef

  * max length of candidate results
  * Range: [topk, 4096]
  * recommended value: 64


NSG
^^^^

**Create index param**


* search_length

  * Range: [10, 300]
  * recommended value: 45


* out_degree

  * Range: [5, 300]
  * recommended value: 50

* candidate_pool_size

  * Range: [50, 1000]
  * recommended value: 300

* knng

  * Range: [5, 300]
  * recommended value: 100

**Search param**

* search_length

  * Range: [10, 300]
  * recommended value: 100


ANNOY
^^^^^^

**Create index param**


* n_trees

  * Range: [1, 1024]


**Search param**

* search_k

  * Range: -1  [topk,  ∞)


