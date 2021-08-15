.. _index:

Index
=========
.. currentmodule:: pymilvus

An index belongs to a specific vector field in a collection, it helps accelerating search.

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+------------------------------+-----------------------------------------------------------------------+
| Constructor                  | Description                                                           |
+==============================+=======================================================================+
| `Index() <#pymilvus.Index>`_ | Create index on a specified column according to the index parameters. |
+------------------------------+-----------------------------------------------------------------------+


Attributes
---------------------
.. autosummary::
   :toctree: api/

+------------------------------------------------------+-----------------------------------------------+
| API                                                  | Description                                   |
+======================================================+===============================================+
| `params <#pymilvus.Index.params>`_                   | Return the index params.                      |
+------------------------------------------------------+-----------------------------------------------+
| `collection_name <#pymilvus.Index.collection_name>`_ | Return corresponding collection name.         |
+------------------------------------------------------+-----------------------------------------------+
| `field_name <#pymilvus.Index.field_name>`_           | Return corresponding field name.              |
+------------------------------------------------------+-----------------------------------------------+
| `drop <#pymilvus.Index.drop>`_                       | Drop index and its corresponding index files. |
+------------------------------------------------------+-----------------------------------------------+


APIs
-----


.. autoclass:: pymilvus.Index
   :member-order: bysource
   :members: params, collection_name, field_name, drop
