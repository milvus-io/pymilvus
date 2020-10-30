==============
API reference
==============

Client
======


Constructor
------------

+----------------------------------------+-----------------------------------------------------------------------+
| Constructor                            | Description                                                           |
+========================================+=======================================================================+
| `Milvus() <#milvus.Milvus>`_           | milvus client                                                         |
+----------------------------------------+-----------------------------------------------------------------------+

Methods
--------

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                  | Description                                                            |
+======================================================================+========================================================================+
| `create_collection() <#milvus.Milvus.create_collection>`_            | Create a collection.                                                   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_collection() <#milvus.Milvus.has_collection>`_                  | Check if collection exists.                                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_collection_info() <#milvus.Milvus.get_collection_info>`_        | Obtain collection information.                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `count_entities() <#milvus.Milvus.count_entities>`_                  | Obtain the number of entity in a collection.                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_collections() <#milvus.Milvus.list_collections>`_              | Get the list of collections.                                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_collection_stats() <#milvus.Milvus.get_collection_stats>`_      | Obtain collection statistics information.                              |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `load_collection() <#milvus.Milvus.load_collection>`_                | Load collection from disk to memory.                                   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_collection() <#milvus.Milvus.drop_collection>`_                | drop a collection.                                                     |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `insert() <#milvus.Milvus.insert>`_                                  | insert entities into specified collection.                             |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_entity_by_id() <#milvus.Milvus.get_entity_by_id>`_              | Obtain entities by providing entity ids.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_id_in_segment() <#milvus.Milvus.list_id_in_segment>`_          | Obtain the list of ids in specified segment.                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `create_index() <#milvus.Milvus.create_index>`_                      | Create an index on specified field.                                    |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_index() <#milvus.Milvus.drop_index>`_                          | Drop index on specified field.                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `create_partition() <#milvus.Milvus.create_partition>`_              | Create a partition under specified collection.                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_partition() <#milvus.Milvus.has_partition>`_                    | Check if specified partition exists under a collection.                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_partitions() <#milvus.Milvus.list_partitions>`_                | Obtain list of partitions under a collection.                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_partition() <#milvus.Milvus.drop_partition>`_                  | Drop specified partition under a collection.                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `search() <#milvus.Milvus.search>`_                                  | Search approximate nearest entities.                                   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `search_in_segment() <#milvus.Milvus.search_in_segment>`_            | Search approximate nearest entities in specified segments.             |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `delete_entity_by_id() <#milvus.Milvus.delete_entity_by_id>`_        | Delete entities by providing entity ids.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `flush() <#milvus.Milvus.flush>`_                                    | Flush collection data from memory to storage.                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `compact() <#milvus.Milvus.compact>`_                                | Compact specified collection.                                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+


APIs
-----


.. autoclass:: milvus.Milvus
   :members: create_collection, has_collection, get_collection_info, count_entities, list_collections,
             get_collection_stats, load_collection, drop_collection, insert, get_entity_by_id, list_id_in_segment,
             create_index, drop_index, create_partition, has_partition, list_partitions, drop_partition,
             search, search_in_segment, delete_entity_by_id, flush, compact




