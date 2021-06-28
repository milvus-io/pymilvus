.. _utility:
    :toctree: api/

Utility
==========

Methods
-------

+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| API                                                                                                                                | Description                                  |
+====================================================================================================================================+==============================================+
| `loading_progress(collection_name, [partition_names,using]) <#pymilvus_orm.utility.loading_progress>`_                             | Query the progress of loading.               |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `wait_for_loading_complete(collection_name, [partition_names, timeout, using]) <#pymilvus_orm.utility.wait_for_loading_complete>`_ | Wait until loading is complete.              |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `index_building_progress(collection_name, [using]) <#pymilvus_orm.utility.index_building_progress>`_                               | Query the progress of index building.        |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `wait_for_index_building_complete(collection_name, [timeout, using]) <#pymilvus_orm.utility.wait_for_index_building_complete>`_    | Wait util index building is complete.        |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `has_collection(collection_name, [using]) <#pymilvus_orm.utility.has_collection>`_                                                 | Check if a specified collection exists.      |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `has_partition(collection_name, partition_name, [using]) <#pymilvus_orm.utility.has_partition>`_                                   | Check if a specified partition exists.       |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+
| `list_collections([timeout, using]) <#pymilvus_orm.utility.list_collections>`_                                                     | List all collections.                        |
+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------+

APIs References
---------------

.. automodule:: pymilvus_orm.utility 
   :member-order: bysource
   :members: loading_progress, wait_for_loading_complete, index_building_progress,
             wait_for_index_building_complete, has_collection, has_partition, list_collections
