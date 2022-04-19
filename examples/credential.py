from pymilvus import utility, connections

_COLLECTION = "demo"

_HOST = '127.0.0.1'
_PORT = '19530'

_ROOT = "root"
_CONNECTION_NAME = "default"
_ANOTHER_CONNECTION_NAME = "another_conn"

_USER = "user"
_PASSWORD = "password"
_ANOTHER_USER = "another_user"
_ANOTHER_PASSWORD = "another_password"
_NEW_PASSWORD = "new_password"


def connect_without_auth(connection_name, host, port):
    print(f"connect to milvus without auth")
    print(f"connection: {connection_name}, host: {host}, port: {port}\n")
    connections.connect(alias=connection_name,
                        host=host,
                        port=port,
                        )


def connect_to_milvus(connection_name, host, port, user, password):
    print(f"connect to milvus user and password")
    print(f"connection: {connection_name}, host: {host}, port: {port}\n")
    connections.connect(alias=connection_name,
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        )


def create_credential(connection_name, user, password):
    print(f"create credential, connection: {connection_name}, user: {user}, password: {password}\n")
    utility.create_credential(user, password, using=connection_name)


def update_credential(connection_name, user, old_password, new_password):
    print(f"update credential, connection: {connection_name}, user: {user}, old_password: {old_password}, new_password: {new_password}\n")
    utility.update_credential(user, old_password, new_password, using=connection_name)


def delete_credential(connection_name, user):
    print(f"delete credential, connection: {connection_name}, user: {user}\n")
    utility.delete_credential(user, using=connection_name)


def reset_password(connection_name, user, old_password, new_password):
    print(f"reset password, connection: {connection_name}, user: {user}, old_password: {old_password}, new_password: {new_password}\n")
    utility.reset_password(user, old_password, new_password, using=connection_name)


def test_connection(connection_name):
    print(f"test for {connection_name}")
    has = utility.has_collection(_COLLECTION, using=connection_name)
    print(f"has collection {_COLLECTION}: {has}")
    users = utility.list_cred_users(using=connection_name)
    print(f"users in Milvus: {users}")
    print(f"test for {connection_name} done\n")


def run():
    connect_without_auth(_ROOT, _HOST, _PORT)
    test_connection(_ROOT)

    # after credential created, _ROOT was not able to call rpc of Milvus.
    # we should use new connection to communicate with Milvus.
    create_credential(_ROOT, _USER, _PASSWORD)
    connect_to_milvus(_CONNECTION_NAME, _HOST, _PORT, _USER, _PASSWORD)
    test_connection(_CONNECTION_NAME)

    create_credential(_CONNECTION_NAME, _ANOTHER_USER, _ANOTHER_PASSWORD)
    update_credential(_CONNECTION_NAME, _ANOTHER_USER, _ANOTHER_PASSWORD, _NEW_PASSWORD)
    connect_to_milvus(_ANOTHER_CONNECTION_NAME, _HOST, _PORT, _ANOTHER_USER, _NEW_PASSWORD)
    test_connection(_ANOTHER_CONNECTION_NAME)

    reset_password(_CONNECTION_NAME, _USER, _PASSWORD, _NEW_PASSWORD)
    test_connection(_CONNECTION_NAME)

    delete_credential(_CONNECTION_NAME, _ANOTHER_USER)
    delete_credential(_CONNECTION_NAME, _USER)
    test_connection(_ROOT)


if __name__ == "__main__":
    run()

