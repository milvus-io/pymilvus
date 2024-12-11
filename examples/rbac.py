import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

super_user = "root"
super_password = "Milvus"

fmt = "\n=== {:30} ===\n"
dim = 8

collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530", user=super_user, password=super_password)

milvus_client.drop_user("user1")
milvus_client.drop_user("user2")
milvus_client.drop_user("user3")

milvus_client.create_user("user1", "password1")
milvus_client.create_user("user2", "password2")
milvus_client.create_user("user3", "password3")

users = milvus_client.list_users()
print("users:", users)

milvus_client.drop_user("user3")

users = milvus_client.list_users()
print("after drop opeartion, users:", users)


db_rw_privileges = [
  {"object_type": "Global", "object_name": "*", "privilege": "CreateCollection"},
  {"object_type": "Global", "object_name": "*", "privilege": "DropCollection"},
  {"object_type": "Global", "object_name": "*", "privilege": "DescribeCollection"},
  {"object_type": "Global", "object_name": "*", "privilege": "ShowCollections"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Search"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Query"},
  {"object_type": "Collection", "object_name": "*", "privilege": "CreateIndex"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Load"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Release"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Delete"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Insert"},
]

db_ro_privileges = [
  {"object_type": "Global", "object_name": "*", "privilege": "DescribeCollection"},
  {"object_type": "Global", "object_name": "*", "privilege": "ShowCollections"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Search"},
  {"object_type": "Collection", "object_name": "*", "privilege": "Query"},
]

role_db_rw = "db_rw"
role_db_ro = "db_ro"
role_custom = "custom_role"
role_cluster_admin = "cluster_admin"
role_database_readonly = "database_readonly"
role_collection_readwrite = "collection_readwrite"

current_roles = milvus_client.list_roles()
print("current roles:", current_roles)

for role in [role_db_rw, role_db_ro]:
    if role in current_roles:
        role_info = milvus_client.describe_role(role)
        for item in role_info['privileges']:
            milvus_client.revoke_privilege(role, item["object_type"], item["privilege"], item["object_name"])
        
        milvus_client.drop_role(role)

# manage custom privilege group and grant it to custom role
privilege_group_name = "custom_privilege_group"
milvus_client.create_privilege_group(privilege_group_name)
milvus_client.add_privileges_to_group(privilege_group_name, ["Search", "Query"])
milvus_client.list_privilege_groups()
milvus_client.remove_privileges_from_group(privilege_group_name, ["Search"])
milvus_client.list_privilege_groups()
milvus_client.create_role(role_custom)
milvus_client.grant_privilege_v2(role_custom, privilege_group_name, "*")

# grant cluster level built-in privilege group
milvus_client.create_role(role_cluster_admin)
milvus_client.grant_privilege_v2(role_cluster_admin, "ClusterAdmin", "*", "*")

# grant database level built-in privilege group
milvus_client.create_role(role_database_readonly)
milvus_client.grant_privilege_v2(role_database_readonly, "DatabaseReadOnly", "*", "db1")

# grant collection level built-in privilege group
milvus_client.create_role(role_collection_readwrite)
milvus_client.grant_privilege_v2(role_collection_readwrite, "CollectionReadWrite", "col1", "db1")

roles = milvus_client.list_roles()
print("roles:", roles)

milvus_client.create_role(role_db_rw)
for item in db_rw_privileges:
    milvus_client.grant_privilege(role_db_rw, item["object_type"], item["privilege"], item["object_name"])


milvus_client.create_role(role_db_ro)
for item in db_ro_privileges:
    milvus_client.grant_privilege(role_db_ro, item["object_type"], item["privilege"], item["object_name"])


roles = milvus_client.list_roles()
print("roles:", roles)
for role in roles:
    role_info = milvus_client.describe_role(role)
    print(f"info for {role}:", role_info)


user1_info = milvus_client.describe_user("user1")
print("user info for user1:", user1_info)
print(f"grant {role_db_rw} to user1")
milvus_client.grant_role("user1", role_db_rw)
print("user info for user1:", user1_info)
milvus_client.grant_role("user1", role_collection_readwrite)

milvus_client.grant_role("user2", role_db_ro)
milvus_client.grant_role("user2", role_db_rw)
milvus_client.grant_role("user2", role_database_readonly)
milvus_client.grant_role("user2", role_cluster_admin)

user2_info = milvus_client.describe_user("user2")
print("user info for user2:", user2_info)
print(f"revoke {role} from user2")
milvus_client.revoke_role("user2", role_db_rw)
user2_info = milvus_client.describe_user("user2")
print("user info for user2:", user2_info)

user3_info = milvus_client.describe_user("user3")
print("user info for user3:", user3_info)


# revoke all privileges before dropping roles and users
milvus_client.revoke_privilege_v2(role_cluster_admin, "ClusterAdmin", "*", "*")
milvus_client.revoke_privilege_v2(role_database_readonly, "DatabaseReadOnly", "*", "db1")
milvus_client.revoke_privilege_v2(role_collection_readwrite, "CollectionReadWrite", "col1", "db1")
milvus_client.revoke_privilege_v2(role_custom, privilege_group_name, "*")

milvus_client.drop_role(role_cluster_admin)
milvus_client.drop_role(role_database_readonly)
milvus_client.drop_role(role_collection_readwrite)
milvus_client.drop_role(role_custom)
milvus_client.list_roles

milvus_client.drop_privilege_group(privilege_group_name)

milvus_client.drop_user("user1")
milvus_client.drop_user("user2")
milvus_client.drop_user("user3")    

