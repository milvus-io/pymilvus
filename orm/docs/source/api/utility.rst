.. _utility:
    :toctree: api/

Utility
==========

Methods
-------

+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                                               | Description                                                            |
+===================================================================================================+========================================================================+
| `loading_progress() <#pymilvus_orm.utility.loading_progress>`_                                    | Drop the collection, as well as its corresponding index files.         |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `wait_for_loading_complete() <#pymilvus_orm.utility.wait_for_loading_complete>`_                  | Load the collection from disk to memory.                               |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `index_building_progress() <#pymilvus_orm.utility.index_building_progress>`_                      | Release the collection from memory.                                    |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `wait_for_index_building_complete() <#pymilvus_orm.utility.wait_for_index_building_complete>`_    | Insert data into collection.                                           |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_collection() <#pymilvus_orm.utility.has_collection>`_                                        | Return the number of entities.                                         |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_partition() <#pymilvus_orm.utility.has_partition>`_                                          | Return the partition corresponding to name.                            |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+
| `list_collections() <#pymilvus_orm.utility.list_collections>`_                                    | Checks if a specified partition exists.                                |
+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------+

APIs References
---------------

.. automodule:: pymilvus_orm.utility 
   :member-order: bysource
   :members: loading_progress, wait_for_loading_complete, index_building_progress,
             wait_for_index_building_complete, has_collection, has_partition, list_collections

