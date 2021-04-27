.. _Connections:

Connections
=========
.. currentmodule:: pymilvus_orm

Connections .

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------------------------+---------------------------------------------------------------------------+
| Constructor                                                          | Description                                                               |
+======================================================================+===========================================================================+
| `Connections() <#pymilvus_orm.Connection>`_                          | Connections is a class which is used to manage all connections of milvus. |
+----------------------------------------------------------------------+---------------------------------------------------------------------------+


Methods
---------------------
.. autosummary::
   :toctree: api/

+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| API                                                                  | Description                                                                                    |
+======================================================================+================================================================================================+
| `configure() <#pymilvus_orm.Connections.configure>`_                 | Configure the milvus connections and then create milvus connections by the passed parameters.  |
+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `add_connection() <#pymilvus_orm.Connections.add_connection>`_       | Add a connection object, it will be passed through as-is.                                      |
+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `remove_connection() <#pymilvus_orm.Connections.remove_connection>`_ | Remove connection from the registry.                                                           |
+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `create_connection() <#pymilvus_orm.Connections.create_collection>`_ | Construct a milvus connection and register it under given alias.                               |
+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `get_connection() <#pymilvus_orm.Connections.get_connection>`_       | Retrieve a milvus connection by alias.                                                         |
+----------------------------------------------------------------------+------------------------------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Connections
   :member-order: bysource
   :members: configure, add_connection, remove_connection, create_connection, get_connection
