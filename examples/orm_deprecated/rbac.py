from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    Role
)

# This example shows how to:
#   1.  connect to Milvus server
#   2.  create a role
#   3.  add user to the role
#   4.  assign the role to the user
#   5.  create privilege group
#   6.  add privileges to the privilege group
#   7.  remove privileges from the privilege group
#   8.  list privilege groups of the role
#   9.  grant the privilege group to the role
#   10. grant a built-in privilege group to the role
#   11. revoke the privilege group from the role
#   12. revoke a built-in privilege group from the role
#   13. drop the role

_HOST = 'localhost'
_USER = 'root'
_PASSWORD = 'Milvus'

# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, user=_USER, password=_PASSWORD)
    print(f"\nList connections:")
    print(connections.list_connections())

def create_privilege_group(role, group_name):
    role.create_privilege_group(group_name)

def add_privileges_to_group(role, group_name, privileges):
    role.add_privileges_to_group(group_name, privileges)

def list_privilege_groups(role):
    return role.list_privilege_groups()

def remove_privileges_from_group(role, group_name, privileges):
    role.remove_privileges_from_group(group_name, privileges)

def drop_privilege_group(role, group_name):
    role.drop_privilege_group(group_name)

def grant_v2(role, privilege, collection_name, db_name):
    role.grant_v2(privilege, collection_name, db_name=db_name)

def revoke_v2(role, privilege, collection_name, db_name):
    role.revoke_v2(privilege, collection_name, db_name=db_name)

def list_grants(role):
    return role.list_grants()

def main():
    # Connect to Milvus server
    create_connection()

    # create role
    role = Role("privilege_group_role")
    role.create()

    # list roles
    utility.list_roles(True)

    # create user
    utility.create_user(user="user1", password="Milvus")

    # add user
    role.add_user("user1")

    # create privilege group
    privilege_group = "search_query"
    create_privilege_group(role, privilege_group)

    # add privileges to group
    add_privileges_to_group(role, privilege_group, ["Search", "Query"])

    # list privilege groups
    groups = list_privilege_groups(role)
    print("List privilege groups: ", groups)

    # remove privileges from group
    remove_privileges_from_group(role, privilege_group, ["Query"])

    # grant custom privielge group to role
    grant_v2(role, privilege_group, "*", db_name="*")

    # grant built-in privielge group to role
    grant_v2(role, "ClusterReadOnly", "*", db_name="*")

    # list grants
    grants =list_grants(role)
    print("List grants: ", grants)

    # revoke custom privielge group from role
    revoke_v2(role, privilege_group, "*", db_name="*")

    # revoke built-in privielge group from role
    revoke_v2(role, "ClusterReadOnly", "*", db_name="*")

    # drop privilege group
    drop_privilege_group(role, privilege_group)

    # drop role
    role.drop()


if __name__ == '__main__':
    main()
