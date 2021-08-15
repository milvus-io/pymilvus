.. _Schema:

Schema
=========
.. currentmodule:: pymilvus

CollectionSchema and FieldSchema.

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------+-----------------------+
| Constructor                                        | Description           |
+====================================================+=======================+
| `CollectionSchema() <#pymilvus.CollectionSchema>`_ | Schema of collection. |
+----------------------------------------------------+-----------------------+
| `FieldSchema() <#pymilvus.FieldSchema>`_           | Schema of field.      |
+----------------------------------------------------+-----------------------+


CollectionSchema Attributes
---------------------------
.. autosummary::
   :toctree: api/

+---------------------------------------------------------------+----------------------------------------------------+
| API                                                           | Description                                        |
+===============================================================+====================================================+
| `fields <#pymilvus.CollectionSchema.fields>`_                 | Return the fields of collection.                   |
+---------------------------------------------------------------+----------------------------------------------------+
| `description <#pymilvus.CollectionSchema.description>`_       | Return the description text about the collection.  |
+---------------------------------------------------------------+----------------------------------------------------+
| `primary_field() <#pymilvus.CollectionSchema.primary_field>`_ | Return the primary key column of collection.       |
+---------------------------------------------------------------+----------------------------------------------------+
| `auto_id() <#pymilvus.CollectionSchema.auto_id>`_             | Return whether the ids is automatically generated. |
+---------------------------------------------------------------+----------------------------------------------------+


APIs
-----

.. autoclass:: pymilvus.CollectionSchema
   :member-order: bysource
   :members: fields, description, primary_field, auto_id


FieldSchema Attributes
----------------------
.. autosummary::
   :toctree: api/

+--------------------------------------------------+-------------------------------------------------+
| API                                              | Description                                     |
+==================================================+=================================================+
| `name <#pymilvus.name>`_                         | Return the name of field.                       |
+--------------------------------------------------+-------------------------------------------------+
| `is_primary <#pymilvus.FieldSchema.is_primary>`_ | Return whether the field is primary key column. |
+--------------------------------------------------+-------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus.FieldSchema
   :member-order: bysource
   :members: name, is_primary


