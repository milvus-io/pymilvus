.. _collection:

Collection
=========
.. currentmodule:: pymilvus_orm

Collection .

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------------------------+-----------------------------------------------------------------------+
| Constructor                                                          | Description                                                           |
+======================================================================+=======================================================================+
| `Collection() <#pymilvus_orm.Collection>`_                           | Milvus client                                                         |
+----------------------------------------------------------------------+-----------------------------------------------------------------------+

Attributes
---------------------
.. autosummary::
   :toctree: api/

+----------------------------------------------------------------------+------------------------------------------------------------------------+
| API                                                                  | Description                                                            |
+======================================================================+========================================================================+
| `Collection.schema <#pymilvus_orm.Collection.schema>`_               | Creates a collection.                                                  |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `Collection.description <#pymilvus_orm.Collection.description>`_     | Gets information of a specified collection.                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `Collection.name <#pymilvus_orm.Collection.name>`_                   | Gets the number of entities in a collection.                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `Collection.is_empty <#pymilvus_orm.Collection.is_empty>`_           | Gets a list of collections.                                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+
| `Collection.num_entities <#pymilvus_orm.Collection.num_entities>`_   | Gets collection statistics.                                            |
+----------------------------------------------------------------------+------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Collection
   :member-order: bysource
   :members: schema, description, name, is_empty, num_entities


Methods
---------------------
.. autosummary::
   :toctree: api/

    Collection.load
    Collection.drop
    Collection.release
    Collection.insert
    Collection.search