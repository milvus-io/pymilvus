.. _index:

Index
=========
.. currentmodule:: pymilvus_orm

An index belongs to a specific vector field in a collection, it helps accelerating search.

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| Constructor                                                          | Description                                                               |
+======================================================================+===========================================================================+
| `Index() <#pymilvus_orm.Index>`_                                     | Create index on a specified column according to the index parameters.     |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+


Attributes
---------------------
.. autosummary::
   :toctree: api/

+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| API                                                                  | Description                                                               |
+======================================================================+===========================================================================+
| `params <#pymilvus_orm.Index.params>`_                               | Return the index params.                                                  |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `collection_name <#pymilvus_orm.Index.collection_name>`_             | Return corresponding collection name.                                     |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `field_name <#pymilvus_orm.Index.field_name>`_                       | Return corresponding field name.                                          |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| `drop <#pymilvus_orm.Index.drop>`_                                   | Drop index and its corresponding index files.                             |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Index
   :member-order: bysource
   :members: params, collection_name, field_name, drop
