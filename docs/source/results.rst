===============
Search results
===============


How to deal with search results
--------------------------------

The invocation of `search()` is like this:

>>> results = client.search('demo', query_vectors, topk)

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


Meanwhile, the results object provide attributes to separately access result id array `id_array` and distance array `distance_array`,
so you can traverse the results like this:

>>> for ids, distances in zip(results.id_array, results.distance_array):
...     for id_, dis_ in zip(ids, distances):
...         print(f"id = {id_}, distance = {dis_}")

.. sectionauthor::
   `Bosszou@milvus <https://github.com/BossZou>`_
