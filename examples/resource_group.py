from pymilvus import utility, connections
from example import *
from role_and_privilege import *

_HOST = '127.0.0.1'
_PORT = '19530'

_CONNECTION_NAME = "default"

_COLLECTION_NAME = 'rg_demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

_USER = "user"
_PASSWD = "123456"
_ROLE = "role"

# Vector parameters
_DIM = 128

# Create a Milvus connection

def create_connection(user, passwd):
    print(f"\nCreate connection...")
    connections.connect(alias=_CONNECTION_NAME,host=_HOST, port=_PORT, user=user, password=passwd)
    print(f"\nList connections:")
    print(connections.list_connections())
    
def create_rbac_user():
    create_credential(_USER, _PASSWD, _CONNECTION_NAME)
    role = Role(_ROLE, using=_CONNECTION_NAME)
    print(f"create role, role_name: {_ROLE}")
    role.create()
    print(f"add user")
    role.add_user(_USER)
    print(f"grant privilege")
    role.grant("Global", "*", "CreateResourceGroup")
    role.grant("Global", "*", "DropResourceGroup")
    role.grant("Global", "*", "DescribeResourceGroup")
    role.grant("Global", "*", "ListResourceGroups")
    role.grant("Global", "*", "TransferNode")
    role.grant("Global", "*", "TransferReplica")
    print(f"list grants")
    print(role.list_grants())
    
def remove_rbac_user():
    create_connection("root", "123456")
    role = Role(_ROLE, using=_CONNECTION_NAME)
    print(f"remove user")
    role.remove_user(_USER)
    role.revoke("Global", "*", "CreateResourceGroup")
    role.revoke("Global", "*", "DropResourceGroup")
    role.revoke("Global", "*", "DescribeResourceGroup")
    role.revoke("Global", "*", "ListResourceGroups")
    role.revoke("Global", "*", "TransferNode")
    role.revoke("Global", "*", "TransferReplica")
    role.drop()
    drop_credential(_USER, _CONNECTION_NAME)


def create_resource_group(name):
    print(f"create resource group: {name}")
    utility.create_resource_group(name, using=_CONNECTION_NAME)


def drop_resource_group(name):
    print(f"drop resource group: {name}")
    utility.drop_resource_group(name, using=_CONNECTION_NAME)


def describe_resource_group(name):
    info = utility.describe_resource_group(name, using=_CONNECTION_NAME)
    print(f"describe resource group: {info}")


def list_resource_groups():
    rgs = utility.list_resource_groups(using=_CONNECTION_NAME)
    print(f"list resource group: {rgs}")


def transfer_node(source, target, num_node):
    print(f"transfer {num_node} nodes from {source} to {target}")
    utility.transfer_node(source, target, num_node, using=_CONNECTION_NAME)


def transfer_replica(source, target, collection_name, num_replica):
    print(
        f"transfer {num_replica} replicas in {collection_name} from {source} to {target}")
    utility.transfer_replica(
        source, target, collection_name, num_replica, using=_CONNECTION_NAME)
    
def load_collection_for_transfer(): 
    coll = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
    vectors = insert(coll, 10000, _DIM)
    coll.flush()
    create_index(coll, _VECTOR_FIELD_NAME)
    load_collection(coll)

def run():
    create_connection("root", "123456")
    # create_rbac_user()
    # create_connection(_USER, _PASSWD)
    # load_collection_for_transfer()
    
    create_resource_group("rg")
    list_resource_groups()
    describe_resource_group("rg")
    transfer_node("__default_resource_group", "rg", 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")

    transfer_node("rg", "__default_resource_group", 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
   
    
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
    # transfer_replica("__default_resource_group", "rg", _COLLECTION_NAME, 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
   
    drop_resource_group("rg")
    
    # release_collection(coll)
    # drop_collection(_COLLECTION_NAME)
    # remove_rbac_user()

if __name__ == "__main__":
    run()
