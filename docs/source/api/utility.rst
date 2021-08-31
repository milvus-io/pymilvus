.. _utility:
    :toctree: api/

Utility
==========

Methods
-------

+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| API                                                                                                                            | Description                                   |
+================================================================================================================================+===============================================+
| `loading_progress(collection_name, [partition_names,using]) <#pymilvus.utility.loading_progress>`_                             | Query the progress of loading.                |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `wait_for_loading_complete(collection_name, [partition_names, timeout, using]) <#pymilvus.utility.wait_for_loading_complete>`_ | Wait until loading is complete.               |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `index_building_progress(collection_name, [using]) <#pymilvus.utility.index_building_progress>`_                               | Query the progress of index building.         |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `wait_for_index_building_complete(collection_name, [timeout, using]) <#pymilvus.utility.wait_for_index_building_complete>`_    | Wait util index building is complete.         |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `has_collection(collection_name, [using]) <#pymilvus.utility.has_collection>`_                                                 | Check if a specified collection exists.       |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `has_partition(collection_name, partition_name, [using]) <#pymilvus.utility.has_partition>`_                                   | Check if a specified partition exists.        |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `list_collections([timeout, using]) <#pymilvus.utility.list_collections>`_                                                     | List all collections.                         |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `drop_collections(collection_name, [timeout, using]) <#pymilvus.utility.drop_collection>`_                                     | Drop a collection by name.                    |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+
| `calc_distance(vectors_left, vectors_right, params, [timeout, using]) <#pymilvus.utility.calc_distance>`_                      | Calculate distance between two vector arrays. |
+--------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------+

APIs References
---------------

.. automodule:: pymilvus.utility 
   :member-order: bysource
   :members: loading_progress, wait_for_loading_complete, index_building_progress,
             wait_for_index_building_complete, has_collection, has_partition, list_collections,
             drop_collection, calc_distance
