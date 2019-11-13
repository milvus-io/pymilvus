pymilvus Search Results
=======================

GuideLines
----------

Module Contents
---------------

.. autoclass:: milvus.client.abstract.TopKQueryResult
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: milvus.client.abstract.RowQueryResult
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: milvus.client.abstract.QueryResult
    :members:
    :undoc-members:
    :show-inheritance:

How to deal with search results
---------------------------------

We provide two ways to deal with results. For examples, when you want to get result which position is (i, j), id and distance can be visited by:
 * ``id = results[i][j].id``, ``distance = results[i][j].distance``
 * ``id = results.id_array[i][j]``, ``distance = results.distance_array[i][j]``

And you can ergodic whole results by:

>>> for row in results:
>>>     for item in row:
>>>         print("id={}, distance={}".format(item.id, item.distance))

or

>>> for id_list, dis_list in zip(results.id_array, results.distance_array):
>>>     for id, dis in zip(id_list, dis_list):
>>>        print("id={}, distance={}".format(id, dis))

We test the two way and record process time. Note that we comment ``print(...)`` sentence when testing. In situation of searching with nq=10000 and topk=1000, the first way cost about 8.88s and second way only 0.37s. If you want your program to run faster, the second way is recommended.