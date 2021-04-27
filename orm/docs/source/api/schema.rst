.. _Schema:

Schema
=========
.. currentmodule:: pymilvus_orm

CollectionSchema and FieldSchema.

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------------------------+---------------------------------------------+
| Constructor                                                          | Description                                 |
+======================================================================+=============================================+
| `CollectionSchema() <#pymilvus_orm.CollectionSchema>`_               | Schema of collection.                       |
+----------------------------------------------------------------------+---------------------------------------------+
| `FieldSchema() <#pymilvus_orm.FieldSchema>`_                         | Schema of field.                            |
+----------------------------------------------------------------------+---------------------------------------------+


CollectionSchema Attributes
---------------------
.. autosummary::
   :toctree: api/

+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| API                                                                  | Description                                                               |
+======================================================================+===========================================================================+
| `fields <#pymilvus_orm.CollectionSchema.fields>`_                    | Return the fields of collection.                                          |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `description <#pymilvus_orm.CollectionSchema.description>`_          | Return the description text about the collection.                         |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `primary_field() <#pymilvus_orm.CollectionSchema.primary_field>`_    | Return the primary key column of collection.                              |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `auto_id() <#pymilvus_orm.CollectionSchema.auto_id>`_                | Return whether the ids is automatically generated.                        |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+


APIs
-----

.. autoclass:: pymilvus_orm.CollectionSchema
   :member-order: bysource
   :members: fields, description, primary_field, auto_id


FieldSchema Attributes
---------------------
.. autosummary::
   :toctree: api/

+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| API                                                                  | Description                                                               |
+======================================================================+===========================================================================+
| `name <#pymilvus_orm.name>`_                                         | Return the name of field.                                                 |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `is_primary <#pymilvus_orm.FieldSchema.is_primary>`_                 | Return whether the field is primary key column.                           |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.FieldSchema
   :member-order: bysource
   :members: name, is_primary


