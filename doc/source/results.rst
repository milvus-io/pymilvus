===============
Search results
===============


How to deal with search results
--------------------------------

The invocation of `search()` is like this:

>>> results = client.search('demo', dsl)

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

You can obtain field values in the results object, but first you need to specify fields you wanted before search:

>>> results = client.search('demo', dsl, fields=['A', 'B']) # specify wanted fields are 'A' and 'B'

then you can obtain them:

>>> value_A = results[i][j].entity.get('A')

or

>>> value_A = getattr(results[i][j].entity, 'A')

or

>>> value_A = results[i][j].entity.value_of_field('A')



