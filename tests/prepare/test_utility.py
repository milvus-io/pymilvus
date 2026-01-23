"""Tests for utility and miscellaneous Prepare methods."""

import pytest
from pymilvus.client.prepare import Prepare
from pymilvus.client.types import ResourceGroupConfig
from pymilvus.exceptions import ParamError


class TestLoadCollectionRequest:
    """Tests for load_collection."""

    def test_load_with_replica_number(self):
        """Test load collection with replica number."""
        req = Prepare.load_collection("test_coll", replica_number=2)
        assert req.collection_name == "test_coll"
        assert req.replica_number == 2

    @pytest.mark.parametrize(
        "param_key,value",
        [
            pytest.param("refresh", True, id="refresh"),
            pytest.param("_refresh", True, id="_refresh"),
            pytest.param("resource_groups", ["rg1", "rg2"], id="resource_groups"),
            pytest.param("_resource_groups", ["rg1"], id="_resource_groups"),
            pytest.param("load_fields", ["field1", "field2"], id="load_fields"),
            pytest.param("_load_fields", ["field1"], id="_load_fields"),
            pytest.param("skip_load_dynamic_field", True, id="skip_dynamic"),
            pytest.param("_skip_load_dynamic_field", True, id="_skip_dynamic"),
            pytest.param("priority", "high", id="priority"),
        ],
    )
    def test_load_with_optional_params(self, param_key, value):
        """Test load collection with various optional parameters."""
        kwargs = {param_key: value}
        req = Prepare.load_collection("test_coll", **kwargs)
        assert req.collection_name == "test_coll"


class TestLoadPartitionsRequest:
    """Tests for load_partitions."""

    def test_load_partitions_basic(self):
        """Test load partitions with basic parameters."""
        req = Prepare.load_partitions(
            "test_coll",
            partition_names=["part1", "part2"],
        )
        assert req.collection_name == "test_coll"
        assert list(req.partition_names) == ["part1", "part2"]

    def test_load_partitions_with_replica(self):
        """Test load partitions with replica number."""
        req = Prepare.load_partitions(
            "test_coll",
            partition_names=["part1"],
            replica_number=2,
        )
        assert req.replica_number == 2

    @pytest.mark.parametrize(
        "param_key,value",
        [
            pytest.param("refresh", True, id="refresh"),
            pytest.param("_refresh", True, id="_refresh"),
            pytest.param("resource_groups", ["rg1"], id="resource_groups"),
            pytest.param("_resource_groups", ["rg1"], id="_resource_groups"),
            pytest.param("load_fields", ["field1"], id="load_fields"),
            pytest.param("_load_fields", ["field1"], id="_load_fields"),
            pytest.param("skip_load_dynamic_field", True, id="skip_dynamic"),
            pytest.param("_skip_load_dynamic_field", True, id="_skip_dynamic"),
            pytest.param("priority", "high", id="priority"),
        ],
    )
    def test_load_partitions_with_optional_params(self, param_key, value):
        """Test load partitions with various optional parameters."""
        kwargs = {param_key: value}
        req = Prepare.load_partitions("test_coll", partition_names=["part1"], **kwargs)
        assert req.collection_name == "test_coll"


class TestManualCompaction:
    """Tests for manual_compaction."""

    def test_compaction_basic(self):
        """Test basic compaction request."""
        req = Prepare.manual_compaction("test_coll", is_clustering=False, is_l0=False)
        assert req.collection_name == "test_coll"

    def test_compaction_clustering(self):
        """Test clustering compaction request."""
        req = Prepare.manual_compaction("test_coll", is_clustering=True, is_l0=False)
        assert req.majorCompaction is True

    def test_compaction_l0(self):
        """Test L0 compaction request."""
        req = Prepare.manual_compaction("test_coll", is_clustering=False, is_l0=True)
        assert req.l0Compaction is True

    def test_compaction_with_collection_id(self):
        """Test compaction with collection ID."""
        req = Prepare.manual_compaction(
            "test_coll", is_clustering=False, is_l0=False, collection_id=12345
        )
        assert req.collectionID == 12345

    def test_compaction_with_target_size(self):
        """Test compaction with target size."""
        req = Prepare.manual_compaction(
            "test_coll", is_clustering=False, is_l0=False, target_size=1024
        )
        assert req.target_size == 1024

    @pytest.mark.parametrize(
        "is_clustering",
        [
            pytest.param(None, id="none"),
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
        ],
    )
    def test_compaction_invalid_is_clustering(self, is_clustering):
        """Test compaction with invalid is_clustering value."""
        with pytest.raises(ParamError, match="is_clustering value"):
            Prepare.manual_compaction("test_coll", is_clustering=is_clustering, is_l0=False)

    @pytest.mark.parametrize(
        "is_l0",
        [
            pytest.param(None, id="none"),
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
        ],
    )
    def test_compaction_invalid_is_l0(self, is_l0):
        """Test compaction with invalid is_l0 value."""
        with pytest.raises(ParamError, match="is_l0 value"):
            Prepare.manual_compaction("test_coll", is_clustering=False, is_l0=is_l0)


class TestGetCompactionState:
    """Tests for get_compaction_state."""

    def test_get_state_basic(self):
        """Test basic get compaction state."""
        req = Prepare.get_compaction_state(12345)
        assert req.compactionID == 12345

    @pytest.mark.parametrize(
        "compaction_id",
        [
            pytest.param(None, id="none"),
            pytest.param("12345", id="string"),
            pytest.param(12.5, id="float"),
        ],
    )
    def test_get_state_invalid_id(self, compaction_id):
        """Test get compaction state with invalid ID."""
        with pytest.raises(ParamError, match="compaction_id value"):
            Prepare.get_compaction_state(compaction_id)


class TestGetCompactionStateWithPlans:
    """Tests for get_compaction_state_with_plans."""

    def test_get_plans_basic(self):
        """Test basic get compaction state with plans."""
        req = Prepare.get_compaction_state_with_plans(12345)
        assert req.compactionID == 12345

    @pytest.mark.parametrize(
        "compaction_id",
        [
            pytest.param(None, id="none"),
            pytest.param("12345", id="string"),
        ],
    )
    def test_get_plans_invalid_id(self, compaction_id):
        """Test get compaction state with plans with invalid ID."""
        with pytest.raises(ParamError, match="compaction_id value"):
            Prepare.get_compaction_state_with_plans(compaction_id)


class TestGetReplicas:
    """Tests for get_replicas."""

    def test_get_replicas_basic(self):
        """Test basic get replicas."""
        req = Prepare.get_replicas(12345)
        assert req.collectionID == 12345
        assert req.with_shard_nodes is True

    @pytest.mark.parametrize(
        "collection_id",
        [
            pytest.param(None, id="none"),
            pytest.param("12345", id="string"),
        ],
    )
    def test_get_replicas_invalid_id(self, collection_id):
        """Test get replicas with invalid collection ID."""
        with pytest.raises(ParamError, match="collection_id value"):
            Prepare.get_replicas(collection_id)


class TestLoadBalanceRequest:
    """Tests for load_balance_request."""

    def test_load_balance_basic(self):
        """Test basic load balance request."""
        req = Prepare.load_balance_request(
            collection_name="test_coll",
            src_node_id=1,
            dst_node_ids=[2, 3],
            sealed_segment_ids=[100, 101],
        )
        assert req.collectionName == "test_coll"
        assert req.src_nodeID == 1
        assert list(req.dst_nodeIDs) == [2, 3]
        assert list(req.sealed_segmentIDs) == [100, 101]


class TestCreateIndexRequest:
    """Tests for create_index_request."""

    def test_create_index_basic(self):
        """Test basic index creation."""
        req = Prepare.create_index_request(
            "test_coll",
            "vector",
            {"index_type": "IVF_FLAT", "metric_type": "L2", "nlist": 128},
        )
        assert req.collection_name == "test_coll"
        assert req.field_name == "vector"

    def test_create_index_with_name(self):
        """Test index creation with index name."""
        req = Prepare.create_index_request(
            "test_coll",
            "vector",
            {"index_type": "IVF_FLAT"},
            index_name="my_index",
        )
        assert req.index_name == "my_index"

    def test_create_index_invalid_dim(self):
        """Test index creation with invalid dim parameter."""
        with pytest.raises(ParamError, match="dim must be of int"):
            Prepare.create_index_request(
                "test_coll",
                "vector",
                {"dim": "invalid"},
            )

    def test_create_index_dim_none(self):
        """Test index creation with None dim."""
        with pytest.raises(ParamError, match="dim must be of int"):
            Prepare.create_index_request(
                "test_coll",
                "vector",
                {"dim": None},
            )

    def test_create_index_filters_none_values(self):
        """Test index creation filters None values."""
        req = Prepare.create_index_request(
            "test_coll",
            "vector",
            {"index_type": "IVF_FLAT", "optional_param": None},
        )
        param_keys = [p.key for p in req.extra_params]
        assert "optional_param" not in param_keys
        assert "index_type" in param_keys


class TestAlterIndexRequest:
    """Tests for alter_index_properties_request and drop_index_properties_request."""

    def test_alter_index_properties(self):
        """Test alter index properties."""
        req = Prepare.alter_index_properties_request(
            "test_coll",
            "my_index",
            {"mmap.enabled": True},
        )
        assert req.collection_name == "test_coll"
        assert req.index_name == "my_index"

    def test_drop_index_properties(self):
        """Test drop index properties."""
        req = Prepare.drop_index_properties_request(
            "test_coll",
            "my_index",
            ["mmap.enabled"],
        )
        assert req.collection_name == "test_coll"
        assert req.index_name == "my_index"
        assert list(req.delete_keys) == ["mmap.enabled"]


class TestResourceGroupRequests:
    """Tests for resource group requests."""

    def test_create_resource_group_with_config(self):
        """Test create resource group with config."""
        config = ResourceGroupConfig(
            requests={"node_num": 2},
            limits={"node_num": 4},
        )
        req = Prepare.create_resource_group("test_rg", config=config)
        assert req.resource_group == "test_rg"

    def test_update_resource_groups(self):
        """Test update resource groups."""
        configs = {
            "rg1": ResourceGroupConfig(requests={"node_num": 2}),
        }
        req = Prepare.update_resource_groups(configs)
        assert req is not None

    def test_transfer_node(self):
        """Test transfer node request."""
        req = Prepare.transfer_node("source_rg", "target_rg", num_node=2)
        assert req.source_resource_group == "source_rg"
        assert req.target_resource_group == "target_rg"
        assert req.num_node == 2

    def test_transfer_replica(self):
        """Test transfer replica request."""
        req = Prepare.transfer_replica("source_rg", "target_rg", "test_coll", num_replica=1)
        assert req.source_resource_group == "source_rg"
        assert req.target_resource_group == "target_rg"
        assert req.collection_name == "test_coll"
        assert req.num_replica == 1


class TestRbacRequests:
    """Tests for RBAC requests."""

    def test_update_password(self):
        """Test update password request."""
        req = Prepare.update_password_request("user", "old_pass", "new_pass")
        assert req.username == "user"

    def test_delete_user_invalid_type(self):
        """Test delete user with non-string user."""
        with pytest.raises(ParamError, match="invalid user"):
            Prepare.delete_user_request(123)

    def test_operate_user_role(self):
        """Test operate user role request."""
        req = Prepare.operate_user_role_request("user", "role", 1)
        assert req.username == "user"
        assert req.role_name == "role"

    @pytest.mark.parametrize(
        "role_name,include_user_info",
        [
            pytest.param("admin", True, id="admin_with_info"),
            pytest.param("reader", False, id="reader_without_info"),
            pytest.param("", False, id="empty_name"),
        ],
    )
    def test_select_role(self, role_name, include_user_info):
        """Test select role request with various parameters."""
        req = Prepare.select_role_request(role_name, include_user_info=include_user_info)
        assert req.role.name == role_name
        assert req.include_user_info is include_user_info

    @pytest.mark.parametrize(
        "user_name,include_role_info",
        [
            pytest.param("testuser", True, id="user_with_info"),
            pytest.param("admin", False, id="admin_without_info"),
            pytest.param("", False, id="empty_name"),
        ],
    )
    def test_select_user(self, user_name, include_role_info):
        """Test select user request with various parameters."""
        req = Prepare.select_user_request(user_name, include_role_info=include_role_info)
        assert req.user.name == user_name
        assert req.include_role_info is include_role_info

    @pytest.mark.parametrize(
        "role_name,privilege,db_name,collection_name",
        [
            pytest.param("admin", "Insert", "default", "test_coll", id="full_params"),
            pytest.param("reader", "Query", "mydb", "data", id="different_values"),
            pytest.param("admin", "Insert", "", "test_coll", id="empty_db"),
        ],
    )
    def test_operate_privilege_v2(self, role_name, privilege, db_name, collection_name):
        """Test operate privilege v2 request with various parameters."""
        req = Prepare.operate_privilege_v2_request(
            role_name=role_name,
            privilege=privilege,
            operate_privilege_type=1,
            db_name=db_name,
            collection_name=collection_name,
        )
        assert req.role.name == role_name

    def test_operate_privilege(self):
        """Test operate privilege request."""
        req = Prepare.operate_privilege_request(
            role_name="admin",
            object="Collection",
            object_name="test_coll",
            privilege="Insert",
            db_name="default",
            operate_privilege_type=1,
        )
        assert req.entity.role.name == "admin"

    @pytest.mark.parametrize(
        "role_name,object_type,object_name,db_name",
        [
            pytest.param("admin", "Collection", "test_coll", "default", id="collection_grant"),
            pytest.param("admin", "", "", "default", id="empty_object"),
            pytest.param("reader", "Database", "mydb", "mydb", id="database_grant"),
        ],
    )
    def test_select_grant(self, role_name, object_type, object_name, db_name):
        """Test select grant request with various parameters."""
        req = Prepare.select_grant_request(
            role_name=role_name,
            object=object_type,
            object_name=object_name,
            db_name=db_name,
        )
        assert req.entity.role.name == role_name

    @pytest.mark.parametrize(
        "group_name,privileges,expected_count",
        [
            pytest.param("my_group", ["Insert", "Query"], 2, id="two_privileges"),
            pytest.param("readers", ["Query"], 1, id="single_privilege"),
            pytest.param("full", ["Insert", "Query", "Delete"], 3, id="three_privileges"),
        ],
    )
    def test_operate_privilege_group(self, group_name, privileges, expected_count):
        """Test operate privilege group request with various parameters."""
        req = Prepare.operate_privilege_group_req(
            privilege_group=group_name,
            privileges=privileges,
            operate_privilege_group_type=1,
        )
        assert req.group_name == group_name
        assert len(req.privileges) == expected_count


class TestRegisterRequest:
    """Tests for register_request."""

    def test_register_with_user_and_host(self):
        """Test register request with user and host."""
        req = Prepare.register_request(user="testuser", host="localhost")
        assert req.client_info.user == "testuser"
        assert req.client_info.host == "localhost"

    def test_register_with_reserved_params(self):
        """Test register request with reserved parameters."""
        req = Prepare.register_request(user=None, host=None, custom_key="custom_value")
        assert "custom_key" in req.client_info.reserved
