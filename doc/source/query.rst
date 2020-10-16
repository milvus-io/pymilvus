==========
Qyery DSL
==========

Like as ElasticSearch, Milvus provides a Query DSL(Domain Specific Language) consisting of two types of clauses to define queries:
Leaf query clauses
Leaf query clauses look for a particular value in a particular field. Currently milvus support `term`, `range`, `vector` queries.
  term: term query matches the entities which corresponding field value are in the specified list. The format is `term: [$value1, $value2, ...]`
  range: range query matches the entities which corresponding field value are in the specified range. The supported range types are:
         "GT", "GTE", "LT", "LTE".
  vector: vector query only take effect on vector fields, and approximately search nearest vectors. The format should be:
            {"topk": $topk, "query": $vectors, "params": {...}, "metric_type": $metric}

Compound query clauses
Currently, milvus only support boolean query, i.e. `bool`. There are three occurrence types:

+--------------------------+--------------------------------------------------------------+
| Occurrence types         | Description                                                  |
+==========================+==============================================================+
| must                     | The clause must appear in matching entities                  |
+--------------------------+--------------------------------------------------------------+
| must_not                 | The clause must not appear in matching entities              |
+--------------------------+--------------------------------------------------------------+
| should                   | At least one query in the clause must in matching entities   |
+--------------------------+--------------------------------------------------------------+