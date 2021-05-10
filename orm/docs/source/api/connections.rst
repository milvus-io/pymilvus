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

+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| API                                                                        | Description                                                                                    |
+============================================================================+================================================================================================+
| `configure() <#pymilvus_orm.Connections.configure>`_                       | Configure the milvus connections and then create milvus connections by the passed parameters.  |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `remove_connection() <#pymilvus_orm.Connections.remove_connection>`_       | Remove connection from the registry.                                                           |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `create_connection() <#pymilvus_orm.Connections.create_collection>`_       | Construct a milvus connection and register it under given alias.                               |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `get_connection() <#pymilvus_orm.Connections.get_connection>`_             | Retrieve a milvus connection by alias.                                                         |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `list_connections() <#pymilvus_orm.Connections.list_connections>`_         | List all connections.                                                                          |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| `get_connection_addr() <#pymilvus_orm.Connections.get_connection_addr>`_   | Get connection configure by alias.                                                             |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Connections
   :member-order: bysource
   :members: configure, remove_connection, create_connection, get_connection, list_connections, get_connection_addr
