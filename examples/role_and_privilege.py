from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus.orm.role import Role

import random
from sklearn import preprocessing

_CONNECTION = "demo"
_FOO_CONNECTION = "foo_connection"
_HOST = '127.0.0.1'
_PORT = '19530'
_ROOT = "root"
_ROOT_PASSWORD = "Milvus"
_COLLECTION_NAME = "foocol2"


def connect_to_milvus(connection=_CONNECTION, user=_ROOT, password=_ROOT_PASSWORD):
    print(f"connect to milvus\n")
    connections.connect(alias=connection,
                        host=_HOST,
                        port=_PORT,
                        user=user,
                        password=password,
                        )


def create_credential(user, password, connection=_CONNECTION):
    print(f"create credential, user: {user}, password: {password}")
    utility.create_user(user, password, using=connection)
    print(f"create credential down\n")


def update_credential(user, password, old_password, connection=_CONNECTION):
    print(f"update credential, user: {user}, password: {password}, old password: {old_password}")
    utility.reset_password(user, old_password, password, using=connection)
    print(f"update credential down\n")


def drop_credential(user, connection=_CONNECTION):
    print(f"drop credential, user: {user}")
    utility.delete_user(user, using=connection)
    print(f"drop credential down\n")


def select_one_user(username, connection=_CONNECTION):
    print(f"select one user, username: {username}")
    roles = utility.list_user(username, True, using=connection)
    print(roles)
    print(f"select one user done\n")


def select_all_user(connection=_CONNECTION):
    print(f"select all user")
    userinfo = utility.list_users(False, using=connection)
    print(userinfo)
    userinfo = utility.list_users(True, using=connection)
    print(userinfo)
    print(f"select all user done\n")


def select_all_role(connection=_CONNECTION):
    print(f"select_all_role")
    roles = utility.list_roles(False, using=connection)
    print(roles)
    roles = utility.list_roles(True, using=connection)
    print(roles)
    print(f"select_all_role done\n")


def has_collection(collection_name, connection=_CONNECTION):
    print(f"has collection, collection_name: {collection_name}")
    has = utility.has_collection("hello_milvus", using=connection)
    print(has)
    print(f"has collection end")


default_dim = 128
default_nb = 1000


def gen_float_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


def gen_float_data(nb, is_normal=False):
    vectors = gen_float_vectors(nb, default_dim, is_normal)
    entities = [
        [i for i in range(nb)],
        [float(i) for i in range(nb)],
        vectors
    ]
    return entities


# rbac I
def rbac_collection(connection=_CONNECTION):
    print(f"rbac_collection")
    default_float_vec_field_name = "float_vector"
    default_fields = [
        FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name=default_float_vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="create_collection_rbac")
    collection = Collection(name=_COLLECTION_NAME, schema=default_schema, using=connection)

    data = gen_float_data(default_nb)
    collection.insert(data)

    is_exception = False
    try:
        collection.drop()
    except Exception as e:
        print(e)
        is_exception = True
    assert is_exception

    collection = Collection(name=_COLLECTION_NAME, schema=default_schema, using=_CONNECTION)
    collection.drop()
    print(f"rbac_collection done")


# rbac II
def rbac_user(username, password, role_name, connection=_CONNECTION):
    update_credential(username, "pfoo1235", password, connection=connection)
    print(select_one_user(username, connection=connection))
    is_exception = False
    try:
        select_all_user(connection=connection)
    except Exception as e:
        print(e)
        is_exception = True
    assert is_exception
    role = Role(role_name, using=_CONNECTION)
    role.grant("User", "*", "SelectUser")
    print(select_all_user(connection))
    role.revoke("User", "*", "SelectUser")


def role_example():
    role_name = "role_test5"
    role = Role(role_name, using=_CONNECTION)
    print(f"create role, role_name: {role_name}")
    role.create()
    print(f"get users")
    role.get_users()
    print(f"select all role")
    print(select_all_role())
    print(f"drop role")
    role.drop()


def associate_users_with_roles_example():
    username = "root"
    role_name = "public"
    role = Role(role_name, using=_CONNECTION)
    print(f"add user")
    role.add_user(username)
    print(f"get users")
    role.get_users()
    print(select_one_user(username))
    print(select_all_user())
    print(f"remove user")
    role.remove_user(username)


def privilege_example():
    print(f"privilege example")

    username = "foo53"
    password = "pfoo123"
    role_name = "general53"
    privilege_create = "CreateCollection"
    privilege_insert = "Insert"
    object_name = _COLLECTION_NAME

    create_credential(username, password)
    role = Role(role_name, using=_CONNECTION)
    print(f"create role, role_name: {role_name}")
    role.create()
    print(f"add user")
    role.add_user(username)
    print(f"grant privilege")
    role.grant("Global", "*", privilege_create)
    role.grant("Collection", object_name, privilege_insert)
    # role.grant("Collection", object_name, "*")
    # role.grant("Collection", "*", privilege_insert)

    print(f"list grants")
    print(role.list_grants())
    print(f"list grant")
    print(role.list_grant("Collection", object_name))

    connect_to_milvus(connection=_FOO_CONNECTION, user=username, password=password)
    has_collection(_COLLECTION_NAME, connection=_FOO_CONNECTION)
    rbac_collection(connection=_FOO_CONNECTION)
    rbac_user(username, password, role_name, connection=_FOO_CONNECTION)

    print(f"revoke privilege")
    role.revoke("Global", "*", privilege_create)
    role.revoke("Collection", object_name, privilege_insert)
    # role.revoke("Collection", object_name, "*")
    # role.revoke("Collection", "*", privilege_insert)
    print(f"remove user")
    role.remove_user(username)
    role.drop()
    drop_credential(username)


def run():
    connect_to_milvus()
    role_example()
    associate_users_with_roles_example()
    privilege_example()


if __name__ == "__main__":
    run()
