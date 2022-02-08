=========
Partition
=========

A partition is a group of entities in one collection with the same label. Entities inserted without a label
will be tagged a default label by milvus.

Partition is managable, which means managing a group of entities with the same label in one collection.

Constructor
-----------

+--------------------------------------+-------------------+
| Constructor                          | Description       |
+======================================+===================+
| `Partition() <#pymilvus.Partition>`_ | Milvus partition. |
+--------------------------------------+-------------------+


Attributes
----------

+----------------------------------------------------+----------------------------------------+
| API                                                | Description                            |
+====================================================+========================================+
| `description <#pymilvus.Partition.description>`_   | Return the description text.           |
+----------------------------------------------------+----------------------------------------+
| `name <#pymilvus.Partition.name>`_                 | Return the partition name.             |
+----------------------------------------------------+----------------------------------------+
| `is_empty <#pymilvus.Partition.is_empty>`_         | Return whether the Partition is empty. |
+----------------------------------------------------+----------------------------------------+
| `num_entities <#pymilvus.Partition.num_entities>`_ | Return the number of entities.         |
+----------------------------------------------------+----------------------------------------+


Methods
---------------------


+--------------------------------------------+--------------------------------------------------------------------------+
| API                                        | Description                                                              |
+============================================+==========================================================================+
| `drop() <#pymilvus.Partition.drop>`_       | Drop the Partition, as well as its corresponding index files.            |
+--------------------------------------------+--------------------------------------------------------------------------+
| `load() <#pymilvus.Partition.load>`_       | Load the Partition from disk to memory.                                  |
+--------------------------------------------+--------------------------------------------------------------------------+
| `release() <#pymilvus.Partition.release>`_ | Release the Partition from memory.                                       |
+--------------------------------------------+--------------------------------------------------------------------------+
| `insert() <#pymilvus.Partition.insert>`_   | Insert data into partition.                                              |
+--------------------------------------------+--------------------------------------------------------------------------+
| `delete() <#pymilvus.Partition.delete>`_   | Delete entities with an expression condition.                            |
+--------------------------------------------+--------------------------------------------------------------------------+
| `search() <#pymilvus.Partition.search>`_   | Vector similarity search with an optional boolean expression as filters. |
+--------------------------------------------+--------------------------------------------------------------------------+
| `query() <#pymilvus.Partition.query>`_     | Query with a set of criteria.                                            |
+--------------------------------------------+--------------------------------------------------------------------------+

API Refereences
---------------


.. autoclass:: pymilvus.Partition
   :member-order: bysource
   :members: description, name, is_empty, num_entities, drop, load, release, insert, search, query, delete

