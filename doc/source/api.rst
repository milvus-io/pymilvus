==============
API reference
==============

Client
======


Constructor
------------

+-----------------------------------------------------------------------------+----------------------------------+
| Constructor                                                                 | Description                      |
+=============================================================================+==================================+
| `Milvus([host, port, handler, pool, **kwargs]) <#milvus.Milvus>`_           | milvus client                    |
+-----------------------------------------------------------------------------+----------------------------------+

Methods
-------

+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| API                                                                                                                       | Description       |
+===========================================================================================================================+===================+
| `create_collection(collection_name, fields[, timeout]) <#milvus.Milvus.create_collection>`_                               |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `has_collection(self, collection_name[, timeout]) <#milvus.Milvus.has_collection>`_                                       |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `get_collection_info(collection_name[, timeout]) <#milvus.Milvus.get_collection_info>`_                                   |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `count_entities(self, collection_name[, timeout]) <#milvus.Milvus.count_entities>`_                                       |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `list_collections([timeout]) <#milvus.Milvus.list_collections>`_                                                          |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `get_collection_stats(collection_name[, timeout]) <#milvus.Milvus.get_collection_stats>`_                                 |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `load_collection(collection_name, timeout=None) <#milvus.Milvus.load_collection>`_                                        |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `drop_collection(collection_name[, timeout]) <#milvus.Milvus.drop_collection>`_                                           |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `insert(collection_name, entities, ids[, partition_tag, params, timeout, **kwargs]) <#milvus.Milvus.insert>`_             |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `get_entity_by_id(collection_name, ids[, fields, timeout]) <#milvus.Milvus.get_entity_by_id>`_                            |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `list_id_in_segment(collection_name, segment_id[, timeout]) <#milvus.Milvus.list_id_in_segment>`_                         |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `create_index(collection_name, field_name, params[, timeout, **kwargs]) <#milvus.Milvus.create_index>`_                   |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `drop_index(collection_name, field_name[, timeout]) <#milvus.Milvus.drop_index>`_                                         |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `create_partition(collection_name, partition_tag[, timeout]) <#milvus.Milvus.create_partition>`_                          |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `has_partition(collection_name, partition_tag[, timeout]) <#milvus.Milvus.has_partition>`_                                |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `list_partitions(collection_name[, timeout]) <#milvus.Milvus.list_partitions>`_                                           |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `drop_partition(collection_name, partition_tag[, timeout]) <#milvus.Milvus.drop_partition>`_                              |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `search(collection_name, dsl[, partition_tags, fields, timeout, **kwargs]) <#milvus.Milvus.search>`_                      |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `search_in_segment(collection_name, segment_ids, dsl[, fields, timeout, **kwargs]) <#milvus.Milvus.search_in_segment>`_   |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `delete_entity_by_id(collection_name, ids[, timeout]) <#milvus.Milvus.delete_entity_by_id>`_                              |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `flush([collection_name_array, timeout, **kwargs]) <#milvus.Milvus.flush>`_                                               |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+
| `compact(collection_name[, threshold, timeout, **kwargs]) <#milvus.Milvus.compact>`_                                      |                   |
+---------------------------------------------------------------------------------------------------------------------------+-------------------+

APIs
-----


.. autoclass:: milvus.Milvus
   :members: create_collection, has_collection, get_collection_info, count_entities, list_collections,
             get_collection_stats, load_collection, drop_collection, insert, get_entity_by_id, list_id_in_segment,
             create_index, drop_index, create_partition, has_partition, list_partitions, drop_partition,
             search, search_in_segment, delete_entity_by_id, flush,compact





