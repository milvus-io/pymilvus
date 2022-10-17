.. _utility:
    :toctree: api/

Utility
==========

Methods
-------

+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| API                                                                                                                            | Description                                             |
+================================================================================================================================+=========================================================+
| `loading_progress(collection_name, [partition_names,using]) <#pymilvus.utility.loading_progress>`_                             | Query the progress of loading.                          |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `wait_for_loading_complete(collection_name, [partition_names, timeout, using]) <#pymilvus.utility.wait_for_loading_complete>`_ | Wait until loading is complete.                         |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `index_building_progress(collection_name, [using]) <#pymilvus.utility.index_building_progress>`_                               | Query the progress of index building.                   |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `wait_for_index_building_complete(collection_name, [timeout, using]) <#pymilvus.utility.wait_for_index_building_complete>`_    | Wait util index building is complete.                   |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `has_collection(collection_name, [using]) <#pymilvus.utility.has_collection>`_                                                 | Check if a specified collection exists.                 |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `has_partition(collection_name, partition_name, [using]) <#pymilvus.utility.has_partition>`_                                   | Check if a specified partition exists.                  |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `list_collections([timeout, using]) <#pymilvus.utility.list_collections>`_                                                     | List all collections.                                   |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `drop_collections(collection_name, [timeout, using]) <#pymilvus.utility.drop_collection>`_                                     | Drop a collection by name.                              |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `get_query_segment_info([timeout, using]) <#pymilvus.utility.get_query_segment_info>`_                                         | Get segments information from query nodes.              |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `mkts_from_hybridts(ts, [milliseconds, delta]) <#pymilvus.utility.mkts_from_hybridts>`_                                        | Generate hybrid timestamp with a known one.             |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `mkts_from_unixtime(timestamp, [milliseconds, delta]) <#pymilvus.utility.mkts_from_unixtime>`_                                 | Generate hybrid timestamp with Unix time.               |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `mkts_from_datetime(d_time, [milliseconds, delta]) <#pymilvus.utility.mkts_from_datetime>`_                                    | Generate hybrid timestamp with datatime.                |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `hybridts_to_unixtime(hybridts) <#pymilvus.utility.hybridts_to_unixtime>`_                                                     | Convert hybrid timestamp to UNIX Epoch time.            |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `hybridts_to_datetime(hybridts, [tz]) <#pymilvus.utility.hybridts_to_datetime>`_                                               | Convert hybrid timestamp to datetime.                   |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `create_alias(collection_name, alias, [timeout, using]) <#pymilvus.utility.create_alias>`_                                     | Specify alias for a collection.                         |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `alter_alias(collection_name, alias, [timeout, using]) <#pymilvus.utility.alter_alias>`_                                       | Change the alias of a collection to another collection. |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
| `drop_alias(alias, [timeout, using]) <#pymilvus.utility.drop_alias>`_                                                          | Delete the alias.                                       |
+--------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------+
APIs References
---------------

.. automodule:: pymilvus.utility
   :member-order: bysource
   :members: loading_progress, wait_for_loading_complete, index_building_progress,
             wait_for_index_building_complete, has_collection, has_partition, list_collections,
             drop_collection, get_query_segment_info,
             mkts_from_hybridts, mkts_from_unixtime, mkts_from_datetime,
             hybridts_to_unixtime, hybridts_to_datetime, create_alias, alter_alias, drop_alias
