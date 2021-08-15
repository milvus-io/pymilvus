===============
Search results
===============


How to deal with search results
--------------------------------

The invocation of `search()` is like this:

>>> import random
>>> dim = 128
>>> nq = 10
>>> query_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
>>> anns_field = "vector field used to search"
>>> search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
>>> limit = 10  # topk
>>> expr = "FieldA > 10"    # filter record whose value on FieldA is less than 10
>>> results = collection.search(query_vectors, anns_field, search_params, limit, expr)

The result object can be used as a 2-D array. `results[i]` (0 <= i < len(results)) represents topk results of i-th query
vector, and `results[i][j]` (0 <= j < len( `results[i]` )) represents j-th result of i-th query vector. To get result id and distance,
you can invoke like this:

>>> id = results[i][j].id
>>> distance = results[i][j].distance

The results object can be iterated, so you can traverse the results with two-level loop:

>>> for raw_result in results:
...     for result in raw_result:
...         id = result.id  # result id
...         distance = result.distance


Meanwhile, the topk results provide attributes to separately access result ids and distances,
so you can traverse the results like this:

>>> for result in results:
...     for id, dis in zip(result.ids, result.distances):
...         print(f"id = {id}, distance = {dis}")

