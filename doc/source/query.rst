==========
Qyery DSL
==========

Like as ElasticSearch, Milvus provides a Query DSL(Domain Specific Language) consisting of two types of clauses to define queries:
Leaf query clauses
Leaf query clauses look for a particular value in a particular field. Currently milvus support `term`, `range`, `vector` queries.
  term: term query matches the entities which corresponding field value are in the specified list. The format is `term: [$value1, $value2, ...]`
  range: range query matches the entities which corresponding field value are in the specified range. The supported range types are:
         "GT"(greater than), "GTE"(greater than or equal), "LT"(less than), "LTE"(less than or equal).
  vector: vector query only take effect on vector fields, and approximately search nearest vectors. The format should be:
            {"topk": $topk, "query": $vectors, "params": {...}, "metric_type": $metric}.
          here, "topk" is the number of approximately nearest top-k entities for each query vector. "query" is a list of query vectors.
          "params" is search parameters, and "metric_type" indicates which distance computation type to be used.

Compound query clauses
Currently, milvus only support boolean query, i.e. `bool`. There are three occurrence types:

+--------------------------+---------------------------------------------------------------------+
| Occurrence types         | Description                                                         |
+==========================+=====================================================================+
| must                     | The clause must appear in matching entities                         |
+--------------------------+---------------------------------------------------------------------+
| must_not                 | The clause must not appear in matching entities                     |
+--------------------------+---------------------------------------------------------------------+
| should                   | At least one query in the clause must appear in matching entities   |
+--------------------------+---------------------------------------------------------------------+

Here is an example:

.. code-block:: ruby
   :linenos:


   dsl = {
       "bool": {
           "must":[
               {
                   "term": {"A": [1, 2, 5]}
               },
               {
                   "range": {"B": {"GT": 1, "LT": 100}}
               },
               {
                   "vector": {
                      "Vec": {"topk": 10, "query": vectors[:1], "metric_type": "L2", "params": {"nprobe": 10}}
                   }
               }
           ]
       }
   }


In this example, client wants to match the results which must be satisfied with:
   1. field "A" value is in the set {1, 2, 5};
   2. field "B" value is in the range of (1, 100)

For each query vector, the results are sorted by distance in descending order.
Note that, a dsl must contain one `vector` query. In current version(milvus v0.11.x, pymilvus v0.3.x), milvus only
support just one `vector` query in a dsl.