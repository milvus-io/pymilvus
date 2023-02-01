from pymilvus import utility, connections, ResourceGroup
from example import *

_HOST = '127.0.0.1'
_PORT = '19530'

_CONNECTION_NAME = "default"

_COLLECTION_NAME = 'rg_demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_DIM = 128

# Create a Milvus connection


def create_connection():
    print(f"\nCreate connection...")
    connections.connect(alias=_CONNECTION_NAME,host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


def create_resource_group(name):
    print(f"create resource group: {name}")
    rg = ResourceGroup(name, using=_CONNECTION_NAME)
    rg.create()


def drop_resource_group(name):
    print(f"drop resource group: {name}")
    rg = ResourceGroup(name, using=_CONNECTION_NAME)
    rg.drop()


def describe_resource_group(name):
    rg = ResourceGroup(name, using=_CONNECTION_NAME)
    info = rg.describe()
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


def run():
    create_connection()
    create_resource_group("rg")
    describe_resource_group("rg")
    transfer_node("__default_resource_group", "rg", 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")

    coll = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
    vectors = insert(coll, 10000, _DIM)
    coll.flush()
    create_index(coll, _VECTOR_FIELD_NAME)

    # load data to memory
    load_collection(coll)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
    transfer_replica("__default_resource_group", "rg", _COLLECTION_NAME, 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
    
    transfer_node("rg", "__default_resource_group", 1)
    describe_resource_group("__default_resource_group")
    describe_resource_group("rg")
    drop_resource_group("rg")
    release_collection(coll)
    drop_collection(_COLLECTION_NAME)
    

if __name__ == "__main__":
    run()
