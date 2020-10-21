
Inspired by ElasticSearch, Milvus provides a Query DSL(Domain Specific Language) consisting of two types of clauses to define queries:


**Leaf query clauses**

Leaf query clauses look for a particular value in a particular field. Currently milvus support `term`, `range`, `vector` queries.

  * term: term query matches the entities whose specified field value are in the specified list. The format is `"term": {"$field": [$value1, $value2, ...]}`


  * range: range query matches the entities whose specified field value are under the specified range. The format is `"range": {"$field": {"$range-type": $value1 ... }}`. The supported range types are:

    - "GT"(greater than)
    - "GTE"(greater than or equal)
    - "LT"(less than)
    - "LTE"(less than or equal).


  * vector: vector query only takes effect on vector fields, and approximately search nearest vectors. The format should be `{"topk": $topk, "query": $vectors, "params": {...}, "metric_type": $metric}`.

    Here, "topk" is the number of approximately nearest top-k entities for each query vector. "query" is a list of query vectors. "params" is search parameters, and "metric_type" indicates which distance computation type to be used.
    "metric_type" is not necessary if it is specified when creating index.

Note that, the `term` and `range` query act as filter queries for `vector` query, they are pre-filtered.

**Compound query clauses**

Currently, milvus only support boolean query, i.e. `bool`. There are three types:

+--------------------------+---------------------------------------------------------------------+
| Occurrence types         | Description                                                         |
+==========================+=====================================================================+
| must                     | The clause must appear in matching entities                         |
+--------------------------+---------------------------------------------------------------------+
| must_not                 | The clause must not appear in matching entities                     |
+--------------------------+---------------------------------------------------------------------+
| should                   | At least one query in the clause must appear in matching entities   |
+--------------------------+---------------------------------------------------------------------+



The three types abide by the follow rules:

  * vector query cannot belong must and must_not. The follow clauses are not permitted:

```json
   {
      "should": {
         "vector": {...}, 
         ...
      }
   }
```

```json
   {
      "must_not": {
         "vector": {...},
         ...
      }
   }
   ```


  * bool query cannot have a should or must_not clause directly. The follow clauses are not permitted:

```json
   {
      "bool": {
         "should": {...}
      }
   }
```

```json
   {
      "bool": {
         "must_not": {...}
      }
   }
```

  * a leaf query cannot combine with compound query in the same clause. The follow clause is not permitted:

```json
   {
      "bool": {
         "must": {...}, 
         "term": {...}
      }
   }
```

  * The whole clause must and can only contain a vector query. The follow clauses are not permitted:


```json
   {
      "bool": {
         "must": {
            "term": {...}, 
            "range": {...}
         }
      }
   }
```


**Examples**

Here is some examples:

* Example 1

```json
   {
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
                      "Vec": {"topk": 10, "query": [[0.1, 0.2]], "metric_type": "L2", "params": {"nprobe": 10}}
                   }
               }
           ]
       }
   }
```

In this example, client wants to match the results which must be satisfied with:
   * field "A" value is in the set {1, 2, 5};
   * field "B" value is in the range of (1, 100)

For each query vector, the results are sorted by distance in descending order.


* Example 2


```json
   {
       "bool": {
          "must": [
             {
                "should": [
                   {
                      "term": {"A": [1, 2, 5]}
                   },
                   {
                      "range": {"B": {"GT": 1, "LT": 100}}
                   }
                ]
             },
             {
                "must": [
                   {
                      "vector": {
                         "Vec": {"topk": 10, "query": [0.1, 0.2], "metric_type": "L2", "params": {"nprobe": 10}}
                      }
                   }
                ]
             }
          ]
       }
   }
```

In this example, client wants to match the results which should be satisfied with one of the follow condition:
   * field "A" value is in the set {1, 2, 5};
   * field "B" value is in the range (1, 100);