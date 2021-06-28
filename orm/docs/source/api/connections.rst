.. _Connections:

Connections
=========
.. currentmodule:: pymilvus_orm

Before connecting to Milvus, the user needs to configure the address and port of the service, and an alias can be assigned to the configuration. The role of `Connections` is to manage the configuration content of each connection and the corresponding connection object.  Using the `Connections` object, users can either configure a single connection to a single service instance, or configure multiple connections to multiple different service instances.  In PyMilvus-ORM, `Connections` is implemented as a singleton class.

Constructor
-----------
.. autosummary::
   :toctree: api/
   :template: autosummaryclass.rst

+----------------------------------------------------------------------+---------------------------------------------------------------------------------+
| Constructor                                                          | Description                                                                     |
+======================================================================+=================================================================================+
| `Connections() <#pymilvus_orm.Connection>`_                          | A singleton class used to manage connections and correspoinding configurations. |
+----------------------------------------------------------------------+---------------------------------------------------------------------------------+


Methods
---------------------
.. autosummary::
   :toctree: api/

+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| API                                                                             | Description                                                       |
+=================================================================================+===================================================================+
| `add_connection() <#pymilvus_orm.Connections.add_connection>`_                  | Configures a connection, including address and port.              |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `remove_connection(alias) <#pymilvus_orm.Connections.remove_connection>`_       | Delete a connection configuration.                                |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `connect([alias]) <#pymilvus_orm.Connections.connect>`_                         | Create a connection object to connect to Milvus.                  |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `disconnect([alias]) <#pymilvus_orm.Connections.disconnect>`_                   | Disconnect from Milvus and close the connection object.           |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `get_connection([alias]) <#pymilvus_orm.Connections.get_connection>`_           | Retrieve a milvus connection by alias.                            |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `list_connections() <#pymilvus_orm.Connections.list_connections>`_              | List all connections.                                             |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+
| `get_connection_addr([alias]) <#pymilvus_orm.Connections.get_connection_addr>`_ | Retrieves connection's configuration by alias.                    |
+---------------------------------------------------------------------------------+-------------------------------------------------------------------+


APIs
-----


.. autoclass:: pymilvus_orm.Connections
   :member-order: bysource
   :members: add_connection, remove_connection, connect, disconnect, get_connection, list_connections, get_connection_addr
