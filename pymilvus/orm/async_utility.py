from .async_connections import connections

def _get_connection(alias: str):
    return connections._fetch_handler(alias)

def get_server_type(using: str = "default"):
    return _get_connection(using).get_server_type()
