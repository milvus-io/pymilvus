======
Future
======


SearchFuture
------------------

Constructor
~~~~~~~~~~~

+--------------------------------------------+----------------+
| Constructor                                | Description    |
+============================================+================+
| `SearchFuture() <#pymilvus.SearchFuture>`_ | Search future. |
+--------------------------------------------+----------------+

Attributes
~~~~~~~~~~

+---------------------------------------------+-------------------------------+
| API                                         | Description                   |
+=============================================+===============================+
| `result() <#pymilvus.SearchFuture.result>`_ | Return the search result.     |
+---------------------------------------------+-------------------------------+
| `cancel() <#pymilvus.SearchFuture.cancel>`_ | Cancel the search request.    |
+---------------------------------------------+-------------------------------+
| `done() <#pymilvus.SearchFuture.done>`_     | Wait for search request done. |
+---------------------------------------------+-------------------------------+


APIs References
~~~~~~~~~~~~~~~


.. autoclass:: pymilvus.SearchFuture
   :member-order: bysource
   :members: result, cancel, done


MutationFuture
--------------

Constructor
~~~~~~~~~~~

+------------------------------------------------+-----------------+
|  Constructor                                   | Description     |
+================================================+=================+
| `MutationFuture() <#pymilvus.MutationFuture>`_ | Mutationfuture. |
+------------------------------------------------+-----------------+

Attributes
~~~~~~~~~~

+-----------------------------------------------+-------------------------------+
| API                                           | Description                   |
+===============================================+===============================+
| `result() <#pymilvus.MutationFuture.result>`_ | Return the insert result.     |
+-----------------------------------------------+-------------------------------+
| `cancel() <#pymilvus.MutationFuture.cancel>`_ | Cancel the insert request.    |
+-----------------------------------------------+-------------------------------+
| `done() <#pymilvus.MutationFuture.done>`_     | Wait for insert request done. |
+-----------------------------------------------+-------------------------------+


APIs References
~~~~~~~~~~~~~~~


.. autoclass:: pymilvus.MutationFuture
   :member-order: bysource
   :members: result, cancel, done
