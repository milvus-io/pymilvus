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
| `Milvus() <#milvus.Milvus>`_           | Milvus client                                                         |
+----------------------------------------+-----------------------------------------------------------------------+

Methods
--------

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                  | Description                                                            |
+======================================================================+========================================================================+
| `create_collection() <#milvus.Milvus.create_collection>`_            | Creates a collection.                                                  |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_collection() <#milvus.Milvus.has_collection>`_                  | Checks if a collection exists.                                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_collection_info() <#milvus.Milvus.get_collection_info>`_        | Gets information of a specified collection.                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `count_entities() <#milvus.Milvus.count_entities>`_                  | Gets the number of entities in a collection.                      |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_collections() <#milvus.Milvus.list_collections>`_              | Gets a list of collections.                                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_collection_stats() <#milvus.Milvus.get_collection_stats>`_      | Gets collection statistics.                                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `load_collection() <#milvus.Milvus.load_collection>`_                | Loads a specified collection from storage to memory.                   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_collection() <#milvus.Milvus.drop_collection>`_                | Removes a collection.                                                    |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `bulk_insert() <#milvus.Milvus.bulk_insert>`_                        | Inserts columnar entities to a specified collection.                   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `insert() <#milvus.Milvus.insert>`_                                  | Inserts entities to a specified collection.                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `get_entity_by_id() <#milvus.Milvus.get_entity_by_id>`_              | Gets entities by entity ID.                                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_id_in_segment() <#milvus.Milvus.list_id_in_segment>`_          | Gets the list of ids in a specified segment.                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `create_index() <#milvus.Milvus.create_index>`_                      | Creates an index for a specified field.                                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_index() <#milvus.Milvus.drop_index>`_                          | Removes index from a specified field.                                    |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `create_partition() <#milvus.Milvus.create_partition>`_              | Creates a partition in a specified collection.                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_partition() <#milvus.Milvus.has_partition>`_                    | Checks if a partition exists in a specified collection.                 |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_partitions() <#milvus.Milvus.list_partitions>`_                | Gets a partition list from a specified collection.                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_partition() <#milvus.Milvus.drop_partition>`_                  | Removes a specified partition from a collection.                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `search() <#milvus.Milvus.search>`_                                  | Searches for approximate nearest entities.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `search_in_segment() <#milvus.Milvus.search_in_segment>`_            | Searches for approximate nearest entities in the specified segments.   |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `delete_entity_by_id() <#milvus.Milvus.delete_entity_by_id>`_        | Deletes entities by entity ID.                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `flush() <#milvus.Milvus.flush>`_                                    | Flushes collection data from memory to storage.                        |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `compact() <#milvus.Milvus.compact>`_                                | Compacts a specified collection.                                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+


APIs
-----


.. autoclass:: milvus.Milvus
   :members: create_collection, has_collection, get_collection_info, count_entities, list_collections,
             get_collection_stats, load_collection, drop_collection, bulk_insert, insert, get_entity_by_id,
             list_id_in_segment, create_index, drop_index, create_partition, has_partition, list_partitions,
             drop_partition, search, search_in_segment, delete_entity_by_id, flush, compact




