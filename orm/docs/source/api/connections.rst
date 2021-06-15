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

+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| API                                                                        | Description                                                                                                      |
+============================================================================+==================================================================================================================+
| `add_connection() <#pymilvus_orm.Connections.add_connection>`_             | Configures the milvus connections and then creates milvus (``connect()``) connections by the passed parameters.  |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `remove_connection() <#pymilvus_orm.Connections.remove_connection>`_       | Remove connection from the registry.                                                                             |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `connect() <#pymilvus_orm.Connections.connect>`_                           |Constructs a milvus connection and register it under given alias.                                                 |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `disconnect() <#pymilvus_orm.Connections.disconnect>`_                     | Disconnects connection from the registry.                                                                        |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `get_connection() <#pymilvus_orm.Connections.get_connection>`_             | Retrieve a milvus connection by alias.                                                                           |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `list_connections() <#pymilvus_orm.Connections.list_connections>`_         | List all connections.                                                                                            |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| `get_connection_addr() <#pymilvus_orm.Connections.get_connection_addr>`_   | Retrieves connection configure by alias.                                                                         |
+----------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Connections
   :member-order: bysource
   :members: add_connection, remove_connection, connect, disconnect, get_connection, list_connections, get_connection_addr
