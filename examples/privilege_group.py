from pymilvus import MilvusClient


uri = "http://localhost:19530"
c = MilvusClient(uri=uri)

c.create_privilege_group(group_name="query_group")
c.add_privileges_to_group(group_name="query_group", privileges=["Query", "Search"])
c.list_privilege_groups()
c.remove_privileges_from_group(group_name="query_group", privileges=["Query", "Search"])
c.drop_privilege_group(group_name="query_group")
