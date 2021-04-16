class Connections(object):
    """
    Connections is a class which is used to manage all connections of milvus.
    """

    def __init__(self):
        """
        Construct a Connections object.
        """
        pass

    def configure(self, **kwargs):
        """
        Configure the milvus connections and then create milvus connections by the passed parameters.

        Example::

            connections.configure(
                default={"host": "localhost", "port": "19530"},
                dev={"host": "localhost", "port": "19531"},
            )
        
        This will create two milvus connections named default and dev.
        """
        pass

    def add_connection(self, alias, conn):
        """
        Add a connection object, it will be passed through as-is.

        :param alias: The name of milvus connection
        :type alias: str

        :param conn: The milvus connection.
        :type conn: class `Milvus`
        """
        pass

    def remove_connection(self, alias):
        """
        Remove connection from the registry. Raises ``KeyError`` if connection
        wasn't found.

        :param alias: The name of milvus connection
        :type alias: str
        """
        pass

    def create_connection(self, alias="default", **kwargs):
        """
        Construct a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type alias: str
        """
        pass

    def get_connection(self, alias):
        """
        Retrieve a milvus connection by alias.

        :param alias: The name of milvus connection
        :type alias: str
        """
        pass

# Singleton Mode in Python

connections = Connections()
configure = connections.configure
add_connection = connections.add_connection
remove_connection = connections.remove_connection
create_connection = connections.create_connection
get_connection = connections.get_connection
