=========
Collection
=========

Collection .

Constructor
-----------

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| Constructor                                                          | Description                                                            |
+======================================================================+========================================================================+
| `Collection() <#pymilvus_orm.Collection>`_                           | Milvus client                                                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+

Attributes
----------

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                  | Description                                                            |
+======================================================================+========================================================================+
| `schema <#pymilvus_orm.Collection.schema>`_                          | Return the schema of collection.                                       |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `description <#pymilvus_orm.Collection.description>`_                | Return the description text about the collection.                      |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `name <#pymilvus_orm.Collection.name>`_                              | Return the collection name.                                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `is_empty <#pymilvus_orm.Collection.is_empty>`_                      | Return whether the collection is empty.                                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `num_entities <#pymilvus_orm.Collection.num_entities>`_              | Return the number of entities.                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `primary_field <#pymilvus_orm.Collection.primary_field>`_            | Return the primary field of collection.                                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `partitions <#pymilvus_orm.Collection.partitions>`_                  | Return all partitions of the collection.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `indexes <#pymilvus_orm.Collection.indexes>`_                        | Return all indexes of the collection.                                  |
+----------------------------------------------------------------------+------------------------------------------------------------------------+


APIs References
---------------


.. autoclass:: pymilvus_orm.Collection
   :member-order: bysource
   :members: schema, description, name, is_empty, num_entities, primary_field, drop, load, release, insert, search,
             partitions, partition, has_partition, drop_partition, indexes, index, create_index, has_index, drop_index


Methods
-------

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                  | Description                                                            |
+======================================================================+========================================================================+
| `drop() <#pymilvus_orm.Collection.drop>`_                            | Drop the collection, as well as its corresponding index files.         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `load() <#pymilvus_orm.Collection.load>`_                            | Load the collection from disk to memory.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `release() <#pymilvus_orm.Collection.release>`_                      | Release the collection from memory.                                    |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `insert() <#pymilvus_orm.Collection.insert>`_                        | Insert data into collection.                                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `search() <#pymilvus_orm.Collection.search>`_                        | Return the number of entities.                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `partition() <#pymilvus_orm.Collection.partition>`_                  | Return the partition corresponding to name.                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_partition() <#pymilvus_orm.Collection.has_partition>`_          | Checks if a specified partition exists.                                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_partition() <#pymilvus_orm.Collection.drop_partition>`_        | Drop the partition and its corresponding index files.                  |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `index() <#pymilvus_orm.Collection.index>`_                          | Return the index corresponding to name.                                |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `create_index() <#pymilvus_orm.Collection.create_index>`_            | Create index on a specified column according to the index parameters.  |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `has_index() <#pymilvus_orm.Collection.has_index>`_                  | Checks whether a specified index exists.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `drop_index() <#pymilvus_orm.Collection.drop_index>`_                | Drop index and its corresponding index files.                          |
+----------------------------------------------------------------------+------------------------------------------------------------------------+


APIs References
---------------


.. autoclass:: pymilvus_orm.Collection
   :member-order: bysource
   :members: drop, load, release, insert, search, partition, has_partition, drop_partition,
             index, create_index, has_index, drop_index
