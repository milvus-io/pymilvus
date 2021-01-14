
Inspired by ElasticSearch, Milvus provides a Query DSL(Domain Specific Language) consisting of two types of clauses to define queries:

## Leaf query clauses

Leaf query clauses look for a particular value in a particular field. Currently milvus support `term`, `range`, `vector` queries.

  * <b>term</b>: term query matches the entities whose specified field value are in the specified list. The format is `"term": {"$field": [$value1, $value2, ...]}`
    or `"term": {"$field": {"values": [$value1, $value2, ...]}}`


  * <b>range</b>: range query matches the entities whose specified field value are under the specified range.
The format is `"range": {"$field": {"$range-type": $value1, ... }}`. The supported range types are:
    - "GT"(greater than)
    - "GTE"(greater than or equal)
    - "LT"(less than)
    - "LTE"(less than or equal).


  * <b>vector</b>: vector query only takes effect on vector fields, and approximately search nearest vectors. The format should be `{"topk": $topk, "query": $vectors, "params": {...}, "metric_type": $metric}`.

    Here, `topk` is the number of approximately nearest top-k entities for each query vector. `query` is a list of query vectors. `params` is search parameters, and `metric_type` indicates which distance computation type to be used.
    `metric_type` is not necessary if it is specified when creating index.

**Note that, the `term` and `range` query act as filter queries for `vector` query, they are pre-filtered.**

## Compound query clauses

Currently, milvus only support boolean query, i.e. `bool`. There are three types:

+---------------+-------------------------------------------------------------------------------+
| Types         | Description                                                                   |
+===============+===============================================================================+
| must          | The clause must appear in matching entities.                                  |
+---------------+-------------------------------------------------------------------------------+
| must_not      | The clause must not appear in matching entities.                              |
+---------------+-------------------------------------------------------------------------------+
| should        | At least one query in the clause must appear in matching entities.            |
+---------------+-------------------------------------------------------------------------------+

## Examples

Here are some examples:

* **Example 1**

   In this example, we demonstrate writing a dsl. Assume that the target collection contains two scalar field named 'A' and 'B',
   and a vector field named 'Vec' with dim 2. We require the result satisfy the conditions as follows:

    1. The value of field 'A' is in [1, 2, 5].
    2. The value of field 'B' is in the range between 1 and 100.

To satisfy condition 1, we need a ``term`` query which is:

```python
term_query = {
    "term": {
        "A": [1, 2, 5]
    }
}
```

To satisfy condition 2, we need a `range` query which is:

```python
range_query = {
    "range": {
        "B": {
            "GT": 1,
            "LT": 100
        }
    }
}
```

Besides, we want to find the results which are top 10 most approximate vectors compared with *[0.1, 0.2]* in the filtered results. So a possible `vector` query is like:

```python
vector_query = {
    "vector": {
        "topk": 10,
        "query": [[0.1, 0.2]],
        "metric_type": "L2",
        "params": {
            "nprobe": 10 # assume "Vec" field has been indexed by 'IVF_FLAT'
        }
    }
}
```

To satisfy the two conditions at the same time, we need a boolean query `must` , under which the *term_query* , *range_query* and *vector_query* lie.

According to the above, we can output the dsl:

```python
dsl = {
    "bool": {
        "must": [term_query, range_query, vector_query]
    }
}
```

The full view is:

```python
{
    "bool": {
        "must":[
            {
                "term": {
                    "A": [1, 2, 5]
                }
            },
            {
                "range": {
                    "B": {
                        "GT": 1, 
                        "LT": 100
                    }
                }
            },
            {
                "vector": {
                   "Vec": {
                       "topk": 10, 
                       "query": [[0.1, 0.2]], 
                       "metric_type": "L2", 
                       "params": {
                           "nprobe": 10
                       }
                   }
                }
            }
        ]
    }
}
```

For each query vector, the results are sorted by distance in descending order.

* **Example 2**

   In this example, we demonstrate writing a dsl with `should` query. Assume that the target collection
   contains two scalar field named `rating` and `release_year`, and a vector field named `embedding`
   with dim 2. We require the result satisfy the conditions as follows:

    1. The rating is high enough **or** recently released, which means `rating` >= 8.0 or 
    `release_year` is 2019 or 2020.

To satisfy condition 1, we will use `should` query including a `range` query and a `term` query
as below.

```python
range_query = {
    "range": {
        "rating": {"gte": 8.0}
    }
},

term_query = {
    "term": {
        "release_year": [2019, 2020]
    }
}

should_query = { "should": [ range_query, term_query ] }
```

This is a typical vector query.

```python
vector_query = {
    "vector": {
        "topk": 10,
        "query": [[0.1, 0.2]],
        "metric_type": "L2",
        "params": {
            "nprobe": 10 # assume "Vec" field has been indexed by 'IVF_FLAT'
        }
    }
}
```

If we want to combine these two querys, the recommended pattern is:

```python
dsl = {
    "bool": {
        "must": [
            { "must": [ should_query ] },
            { "must": [ vector_query ] }
        ]
    }
}

```

The full view is:

```python
{
    "bool": {
        "must": [
            {
                "must": [
                    {
                         "should": [
                            {
                                "range": {
                                    "rating": {"gte": 8.0}
                                }
                            },
                            {
                                "term": {
                                    "release_year": [2019, 2020]
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "must": [
                    {
                        "vector": {
                            "embedding": {
                                "topk": 10,
                                "query": [[0.1, 0.2]],
                                "metric_type": "L2",
                                "params": {
                                    "nprobe": 10
                                }
                            }
                        }
                    }
                ]
            }
        ]
    }
}
```
## Constraints

.. caution::
   The dsl clause abide by the follow rules.


* `vector` query cannot belong `should` and `must_not`. The follow clauses are not permitted:

```python
   # This is an invalid clause because `vector` is under `should`
   {
      "should": [
         "vector": {...}, 
         ...
      ]
   }
```

```python
   # This is an invalid clause because `vector` is under `must_not`
   {
      "must_not": [
         "vector": {...},
         ...
      ]
   }
```


  * `bool` query cannot have a `should` or `must_not` clause directly. The follow clauses are not permitted:

```python
   # This is an invalid clause because `should` is under `bool`
   {
      "bool": {
         "should": [...]
      }
   }
```

```python
   # This is an invalid clause because `must_not` is under `bool`
   {
      "bool": {
         "must_not": [...]
      }
   }
```

  * A leaf query cannot combine with compound query in the same clause. The follow clause is not permitted:

```python
   # This is an invalid clause because `must` is side by side with `term`
   {
      "bool": {
         "must": [...], 
         "term": {...}
      }
   }
```

  * The whole clause must and can only contain a `vector` query. The follow clauses are not permitted:

```python
   # This is an invalid clause because `vector` not exists.
   {
      "bool": {
         "must": [
            {
                "term": {...}
            },
            {
                "range": {...}
            }
         ]
      }
   }
```

```python
# This is an invalid clause because there are `vector` queries.
{
    "bool": {
        "must": [
            {
                "vector": {...}
            },
            {
                "vector": {...}
            }
        ]
    }
}
```

