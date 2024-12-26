==========
Collection
==========

The scheme of a collection is fixed when collection created. Collection scheme consists of many fields,
and must contain a vector field. A field to collection is like a column to RDBMS table. Data type are the same in one field.

A collection is a set of entities, which are also called rows. An entity contains data of all fields.
Each entity can be labeled, a group of entities with the same label is called a partition. Entity without a
label will be tagged a default label by Milvus.

Constructor
-----------

+----------------------------------------+---------------+
| Constructor                            | Description   |
+========================================+===============+
| `Collection() <#pymilvus.Collection>`_ | Milvus client |
+----------------------------------------+---------------+

Attributes
----------

+-------------------------------------------------------+---------------------------------------------------+
| Attributes                                            | Description                                       |
+=======================================================+===================================================+
| `schema <#pymilvus.Collection.schema>`_               | Return the schema of collection.                  |
+-------------------------------------------------------+---------------------------------------------------+
| `description <#pymilvus.Collection.description>`_     | Return the description text about the collection. |
+-------------------------------------------------------+---------------------------------------------------+
| `name <#pymilvus.Collection.name>`_                   | Return the collection name.                       |
+-------------------------------------------------------+---------------------------------------------------+
| `is_empty <#pymilvus.Collection.is_empty>`_           | Return whether the collection is empty.           |
+-------------------------------------------------------+---------------------------------------------------+
| `num_entities <#pymilvus.Collection.num_entities>`_   | Return the number of entities.                    |
+-------------------------------------------------------+---------------------------------------------------+
| `primary_field <#pymilvus.Collection.primary_field>`_ | Return the primary field of collection.           |
+-------------------------------------------------------+---------------------------------------------------+
| `partitions <#pymilvus.Collection.partitions>`_       | Return all partitions of the collection.          |
+-------------------------------------------------------+---------------------------------------------------+
| `indexes <#pymilvus.Collection.indexes>`_             | Return all indexes of the collection.             |
+-------------------------------------------------------+---------------------------------------------------+



Methods
-------

+---------------------------------------------------------------+--------------------------------------------------------------------------+
| API                                                           | Description                                                              |
+===============================================================+==========================================================================+
| `drop() <#pymilvus.Collection.drop>`_                         | Drop the collection, as well as its corresponding index files.           |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `load() <#pymilvus.Collection.load>`_                         | Load the collection from disk to memory.                                 |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `release() <#pymilvus.Collection.release>`_                   | Release the collection from memory.                                      |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `insert() <#pymilvus.Collection.insert>`_                     | Insert data into collection.                                             |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `delete() <#pymilvus.Collection.delete>`_                     | Delete entities with an expression condition.                            |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `search() <#pymilvus.Collection.search>`_                     | Vector similarity search with an optional boolean expression as filters. |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `upsert() <#pymilvus.Collection.upsert>`_                     | Upsert data of collection.                                               |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `query() <#pymilvus.Collection.query>`_                       | Query with a set of criteria.                                            |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `partition() <#pymilvus.Collection.partition>`_               | Return the partition corresponding to name.                              |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `create_partition() <#pymilvus.Collection.create_partition>`_ | Create the partition for the collection.                                 |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `has_partition() <#pymilvus.Collection.has_partition>`_       | Checks if a specified partition exists.                                  |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `drop_partition() <#pymilvus.Collection.drop_partition>`_     | Drop the partition and its corresponding index files.                    |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `index() <#pymilvus.Collection.index>`_                       | Return the index corresponding to name.                                  |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `create_index() <#pymilvus.Collection.create_index>`_         | Create index on a specified column according to the index parameters.    |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `has_index() <#pymilvus.Collection.has_index>`_               | Checks whether a specified index exists.                                 |
+---------------------------------------------------------------+--------------------------------------------------------------------------+
| `drop_index() <#pymilvus.Collection.drop_index>`_             | Drop index and its corresponding index files.                            |
+---------------------------------------------------------------+--------------------------------------------------------------------------+


APIs References
---------------

.. autoclass:: pymilvus.Collection
   :member-order: bysource
   :special-members: __init__
   :members: schema, description, name, is_empty, num_entities, primary_field, partitions, indexes,
             drop, load, release, insert, delete, search, query, partition, create_partition, has_partition, drop_partition,
             index, create_index, has_index, drop_index
