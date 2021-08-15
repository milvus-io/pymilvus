============
SearchResult
============

SearchResult
------------

Constructor
~~~~~~~~~~~

+--------------------------------------------+---------------+
| Constructor                                | Description   |
+============================================+===============+
| `SearchResult() <#pymilvus.SearchResult>`_ | Search Result |
+--------------------------------------------+---------------+

Attributes
~~~~~~~~~~

+--------------------------------------------------------+-------------------------------------------------+
| API                                                    | Description                                     |
+========================================================+=================================================+
| `iter(self) <#pymilvus.SearchResult.\_\_iter\_\_>`_    | Iterate the search result.                      |
+--------------------------------------------------------+-------------------------------------------------+
| `self[item] <#pymilvus.SearchResult.\_\_getitem\_\_>`_ | Return the Hits corresponding to the nth query. |
+--------------------------------------------------------+-------------------------------------------------+
| `len(self) <#pymilvus.SearchResult.\_\_len\_\_>`_      | Return the number of query of search result.    |
+--------------------------------------------------------+-------------------------------------------------+


APIs References
~~~~~~~~~~~~~~~


.. autoclass:: pymilvus.SearchResult
   :member-order: bysource
   :members: __iter__, __getitem__, __len__


Hits
----

Constructor
~~~~~~~~~~~

+----------------------------+-------------------------------------+
| Constructor                | Description                         |
+============================+=====================================+
| `Hits() <#pymilvus.Hits>`_ | Search result about specific query. |
+----------------------------+-------------------------------------+

Attributes
~~~~~~~~~~

+------------------------------------------------+--------------------------------------+
| API                                            | Description                          |
+================================================+======================================+
| `iter(self) <#pymilvus.Hits.\_\_iter\_\_>`_    | Iterate the hits object.             |
+------------------------------------------------+--------------------------------------+
| `self[item] <#pymilvus.Hits.\_\_getitem\_\_>`_ | Return the hit record to the query.  |
+------------------------------------------------+--------------------------------------+
| `len(self) <#pymilvus.Hits.\_\_len\_\_>`_      | Return the number of hit records.    |
+------------------------------------------------+--------------------------------------+
| `ids <#pymilvus.Hits.ids>`_                    | Return the ids of hit records.       |
+------------------------------------------------+--------------------------------------+
| `distances <#pymilvus.Hits.distances>`_        | Return the distances of hit records. |
+------------------------------------------------+--------------------------------------+


APIs References
~~~~~~~~~~~~~~~


.. autoclass:: pymilvus.Hits
   :member-order: bysource
   :members: __iter__, __getitem__, __len__, ids, distances


Hit
---

Constructor
~~~~~~~~~~~

+--------------------------+-------------------------------------+
| Constructor              | Description                         |
+==========================+=====================================+
| `Hit() <#pymilvus.Hit>`_ | Search result about specific query. |
+--------------------------+-------------------------------------+

Attributes
~~~~~~~~~~

+------------------------------------------+---------------------------------------+
| API                                      | Description                           |
+==========================================+=======================================+
| `id <#pymilvus.Hit.id>`_                 | Return the id of hit record.          |
+------------------------------------------+---------------------------------------+
| `distance <#pymilvus.Hit.distance>`_     | Return the distance of hit record.    |
+------------------------------------------+---------------------------------------+
| `score <#pymilvus.Hit.score>`_           | Return the score of hit record.       |
+------------------------------------------+---------------------------------------+
| `str(self) <#pymilvus.Hit.\_\_str\_\_>`_ | Return the information of hit record. |
+------------------------------------------+---------------------------------------+


APIs References
~~~~~~~~~~~~~~~


.. autoclass:: pymilvus.Hit
   :member-order: bysource
   :members: id, distance, score, __str__

