==========
Qyery DSL
==========

Like as ElasticSearch, Milvus provides a Query DSL(Domain Specific Language) consisting of two types of clauses to define queries:
Leaf query clauses
Leaf query clauses look for a particular value in a particular field. Currently milvus support `term`, `range`, `vector` queries.
  term: term query macths the entities which corresponding field value are in the specified list.

Compound query clauses
Currently, milvus only support boolean query, i.e. `bool`. There are three occurrence types:

+--------------------------+----------------------------------+
| Occurrence types         | Description                      |
+==========================+==================================+
| must                     | milvus client                    |
+--------------------------+----------------------------------+
| must                     | milvus client                    |
+--------------------------+----------------------------------+