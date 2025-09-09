from pymilvus import utility, connections, Collection
from pymilvus.client.constants import DEFAULT_RESOURCE_GROUP
from pymilvus.client.types import ResourceGroupConfig
from typing import List
from example import create_connection, create_collection, insert, create_index

_PENDING_NODES_RESOURCE_GROUP="pending_nodes"
# Vector parameters
_DIM = 128
_COLLECTION_NAME = 'rg_declarative_demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

def create_example_collection_and_load(replica_number: int, resource_groups: List[str]):
    print(f"\nCreate collection and load...")
    coll = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
    insert(coll, 10000, _DIM)
    coll.flush()
    create_index(coll, _VECTOR_FIELD_NAME)
    coll.load(replica_number=replica_number, _resource_groups=resource_groups)

def transfer_replica(src: str, dest: str, num_replica: int):
    utility.transfer_replica(source_group=src, target_group=dest, collection_name=_COLLECTION_NAME, num_replicas=num_replica)

def list_replica():
    coll = Collection(name=_COLLECTION_NAME)
    replicas = coll.get_replicas()
    print(replicas)

def init_cluster(node_num: int):
    print(f"Init cluster with {node_num} nodes, all nodes will be put in default resource group")
    # create a pending resource group, which can used to hold the pending nodes that do not hold any data.
    utility.create_resource_group(name=_PENDING_NODES_RESOURCE_GROUP, config=ResourceGroupConfig(
        requests={"node_num": 0}, # this resource group can hold 0 nodes, no data will be load on it.
        limits={"node_num": 10000}, # this resource group can hold at most 10000 nodes 
    ))

    # create a default resource group, which can used to hold the nodes that all initial node in it.
    utility.update_resource_groups({
        DEFAULT_RESOURCE_GROUP: ResourceGroupConfig(
            requests={"node_num": node_num},
            limits={"node_num": node_num},
            transfer_from=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover missing node from pending resource group at high priority.
            transfer_to=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover redundant node to pending resource group at low priority.
        )})

def list_all_resource_groups():
    rg_names = utility.list_resource_groups()

    for rg_name in rg_names:
        resource_group = utility.describe_resource_group(rg_name)
        print(resource_group)
        # print(f"Resource group {rg_name} has {resource_group.nodes} with config: {resource_group.config}")

def scale_resource_group_to(name :str, node_num: int):
    """scale resource group to node_num nodes, new query node need to be added from outside orchestration system"""
    utility.update_resource_groups({
        name: ResourceGroupConfig(
            requests={"node_num": node_num},
            limits={"node_num": node_num},
            transfer_from=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover missing node from pending resource group at high priority.
            transfer_to=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover redundant node to pending resource group at low priority.
        )
    })

def create_resource_group(name: str, node_num: int):
    print(f"Create resource group {name} with {node_num} nodes")
    utility.create_resource_group(name, config=ResourceGroupConfig(
        requests={"node_num": node_num},
        limits={"node_num": node_num},
        transfer_from=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover missing node from pending resource group at high priority.
        transfer_to=[{"resource_group": _PENDING_NODES_RESOURCE_GROUP}], # recover redundant node to pending resource group at low priority.
    ))

def resource_group_management():
    # cluster is initialized with 1 node in default resource group, and 0 node in pending resource group.
    init_cluster(1)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 0

    # rg1 missing two query node.
    create_resource_group("rg1", 2)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 0
    # rg1: 0(missing 2)

    # scale_out(2)
    # scale out two new query node into cluster by orchestration system, these node will be added to rg1 automatically.
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 0
    # rg1: 2


    # rg1 missing one query node.
    scale_resource_group_to("rg1", 3)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 0
    # rg1: 2(missing 1)

    # scale_out(2)
    # scale out two new query node into cluster by orchestration system, one node will be added to rg1 automatically
    # and one redundant node will be added to pending resource group.
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 1
    # rg1: 3

    scale_resource_group_to("rg1", 1)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 3
    # rg1: 1

    # rg2 missing three query node, will be added from pending resource group.
    create_resource_group("rg2", 3)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1
    # _PENDING_NODES_RESOURCE_GROUP: 0
    # rg1: 1
    # rg2: 3

    scale_resource_group_to(DEFAULT_RESOURCE_GROUP, 5)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 1(missing 4)
    # _PENDING_NODES_RESOURCE_GROUP: 0
    # rg1: 1
    # rg2: 3

    # scale_out(4)
    list_all_resource_groups()
    # DEFAULT_RESOURCE_GROUP: 5
    # _PENDING_NODES_RESOURCE_GROUP: 1
    # rg1: 1
    # rg2: 3

def replica_management():
    # load collection into default.
    # create_example_collection_and_load(4, ["rg1", "rg2", "rg2", "rg2"])
    # one replica per node in default resource group.
    list_replica()
    transfer_replica("rg1", DEFAULT_RESOURCE_GROUP, 1)
    list_replica()
    transfer_replica("rg2", DEFAULT_RESOURCE_GROUP, 1)
    list_replica()
    # DEFAULT_RESOURCE_GROUP: 2 replica on 5 nodes.
    # rg1: 0 replica.
    # rg2: 2 replica on 3 nodes.

if __name__ == "__main__":
    create_connection()
    resource_group_management()
    create_example_collection_and_load(4, ["rg1", "rg2", "rg2", "rg2"])
    replica_management()
