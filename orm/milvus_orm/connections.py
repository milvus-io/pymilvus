class Connections(object):

    def __init__(self):
        pass

    def configure(self, **kwargs):
        pass

    def add_connection(self, alias, conn):
        pass

    def remove_connection(self, alias):
        pass

    def create_connection(self, alias="default", **kwargs):
        pass

    def get_connection(self, alias):
        pass

# Singleton Mode in Python

connections = Connections()
configure = connections.configure
add_connection = connections.add_connection
remove_connection = connections.remove_connection
create_connection = connections.create_connection
get_connection = connections.get_connection
