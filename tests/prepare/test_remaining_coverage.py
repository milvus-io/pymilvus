"""Additional tests to reach 90% coverage for Prepare methods."""

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.prepare import Prepare


class TestRowParseEdgeCases:
    """Tests for edge cases in row parsing."""

    def test_parse_row_with_nullable_and_none_value(self):
        """Test parsing row with nullable field and None value."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("nullable", DataType.VARCHAR, nullable=True, max_length=100),
            ]
        )
        rows = [
            {"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "nullable": None},
            {"pk": 2, "vector": [5.0, 6.0, 7.0, 8.0], "nullable": "value"},
        ]
        req = Prepare.row_insert_param(
            "test_coll", rows, "", fields_info=schema.to_dict()["fields"]
        )
        assert req.num_rows == 2


class TestBatchParseEdgeCases:
    """Tests for edge cases in batch parsing."""

    def test_batch_insert_with_valid_data(self):
        """Test batch insert with valid data."""
        entities = [
            {"name": "pk", "values": [1, 2], "type": DataType.INT64},
            {"name": "vector", "values": [[1.0, 2.0], [3.0, 4.0]], "type": DataType.FLOAT_VECTOR},
        ]
        fields_info = [
            {"name": "pk", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
        ]
        req = Prepare.batch_insert_param("coll", entities, "part", fields_info)
        assert req.num_rows == 2


class TestSearchIteratorParams:
    """Tests for search iterator parameters."""

    @pytest.fixture
    def basic_params(self):
        return {"metric_type": "L2", "params": {"nprobe": 10}}

    def test_search_with_all_iterator_params(self, basic_params):
        """Test search with multiple iterator parameters."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_params,
            limit=10,
            is_iterator=True,
            iter_search_v2=True,
            iter_search_batch_size=100,
            iter_search_last_bound=0.5,
            iter_search_id="test_id",
        )
        assert req is not None


class TestStructDataStructures:
    """Tests for struct data structure setup."""

    def test_setup_empty_struct_fields(self):
        """Test setup with empty struct fields."""
        result = Prepare._setup_struct_data_structures(None)
        assert result[0] == {}  # struct_fields_data
        assert result[1] == {}  # struct_info_map
        assert result[2] == {}  # struct_sub_fields_data
        assert result[3] == {}  # struct_sub_field_info
        assert result[4] == []  # input_struct_field_info

    def test_setup_with_struct_fields(self):
        """Test setup with struct fields."""
        struct_fields_info = [
            {
                "name": "metadata",
                "type": DataType.STRUCT,
                "fields": [
                    {"name": "score", "type": DataType.FLOAT},
                ],
            }
        ]
        result = Prepare._setup_struct_data_structures(struct_fields_info)
        assert "metadata" in result[0]  # struct_fields_data
        assert "metadata" in result[1]  # struct_info_map


class TestHelperMethods:
    """Tests for helper methods."""

    def test_is_input_field_auto_id(self):
        """Test _is_input_field with auto_id field."""
        field = {"name": "pk", "auto_id": True}
        # For insert (is_upsert=False), auto_id fields are excluded
        assert Prepare._is_input_field(field, is_upsert=False) is False
        # For upsert (is_upsert=True), auto_id fields are included
        assert Prepare._is_input_field(field, is_upsert=True) is True

    def test_is_input_field_function_output(self):
        """Test _is_input_field with function output field."""
        field = {"name": "embedding", "is_function_output": True}
        assert Prepare._is_input_field(field, is_upsert=False) is False
        assert Prepare._is_input_field(field, is_upsert=True) is False

    def test_function_output_field_names(self):
        """Test _function_output_field_names."""
        fields_info = [
            {"name": "pk", "type": DataType.INT64},
            {"name": "text", "type": DataType.VARCHAR},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "is_function_output": True},
        ]
        result = Prepare._function_output_field_names(fields_info)
        assert result == ["embedding"]

    def test_num_input_fields(self):
        """Test _num_input_fields."""
        fields_info = [
            {"name": "pk", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR},
            {"name": "auto", "type": DataType.INT64, "auto_id": True},
        ]
        # For insert, auto_id field is excluded
        assert Prepare._num_input_fields(fields_info, is_upsert=False) == 2
        # For upsert, auto_id field is included
        assert Prepare._num_input_fields(fields_info, is_upsert=True) == 3


class TestDatabaseRequests:
    """Tests for database requests."""

    def test_alter_database_properties(self):
        """Test alter database properties request."""
        req = Prepare.alter_database_properties_req("test_db", {"prop1": "value1"})
        assert req.db_name == "test_db"

    def test_drop_database_properties(self):
        """Test drop database properties request."""
        req = Prepare.drop_database_properties_req("test_db", ["prop1", "prop2"])
        assert req.db_name == "test_db"
        assert list(req.delete_keys) == ["prop1", "prop2"]


class TestAlterCollectionRequest:
    """Tests for alter collection request."""

    def test_alter_collection_properties(self):
        """Test alter collection properties request."""
        req = Prepare.alter_collection_request("test_coll", {"prop1": "value1"})
        assert req.collection_name == "test_coll"


class TestDropCollectionRequest:
    """Tests for drop collection request."""

    def test_drop_collection(self):
        """Test drop collection request."""
        req = Prepare.drop_collection_request("test_coll")
        assert req.collection_name == "test_coll"


class TestTruncateCollectionRequest:
    """Tests for truncate collection request."""

    def test_truncate_collection(self):
        """Test truncate collection request."""
        req = Prepare.truncate_collection_request("test_coll")
        assert req.collection_name == "test_coll"


class TestListDatabaseRequest:
    """Tests for list database request."""

    def test_list_database(self):
        """Test list database request."""
        req = Prepare.list_database_req()
        assert req is not None


class TestDescribeDatabaseRequest:
    """Tests for describe database request."""

    def test_describe_database(self):
        """Test describe database request."""
        req = Prepare.describe_database_req("test_db")
        assert req.db_name == "test_db"


class TestCreateDatabaseRequest:
    """Tests for create database request."""

    def test_create_database_with_properties(self):
        """Test create database with properties."""
        req = Prepare.create_database_req("test_db", {"replica.number": "2"})
        assert req.db_name == "test_db"


class TestDropDatabaseRequest:
    """Tests for drop database request."""

    def test_drop_database(self):
        """Test drop database request."""
        req = Prepare.drop_database_req("test_db")
        assert req.db_name == "test_db"


class TestUserRequests:
    """Tests for user requests."""

    def test_list_usernames(self):
        """Test list usernames request."""
        req = Prepare.list_usernames_request()
        assert req is not None


class TestRoleRequests:
    """Tests for role requests."""

    def test_create_role(self):
        """Test create role request."""
        req = Prepare.create_role_request("test_role")
        assert req.entity.name == "test_role"

    def test_drop_role(self):
        """Test drop role request."""
        req = Prepare.drop_role_request("test_role")
        assert req.role_name == "test_role"

    def test_drop_role_force(self):
        """Test drop role with force."""
        req = Prepare.drop_role_request("test_role", force_drop=True)
        assert req.role_name == "test_role"
        assert req.force_drop is True


class TestPrivilegeGroupRequests:
    """Tests for privilege group requests."""

    def test_create_privilege_group(self):
        """Test create privilege group request."""
        req = Prepare.create_privilege_group_req("test_group")
        assert req.group_name == "test_group"

    def test_drop_privilege_group(self):
        """Test drop privilege group request."""
        req = Prepare.drop_privilege_group_req("test_group")
        assert req.group_name == "test_group"

    def test_list_privilege_groups(self):
        """Test list privilege groups request."""
        req = Prepare.list_privilege_groups_req()
        assert req is not None


class TestResourceGroupRequestsExtra:
    """Additional tests for resource group requests."""

    def test_list_resource_groups(self):
        """Test list resource groups request."""
        req = Prepare.list_resource_groups()
        assert req is not None

    def test_describe_resource_group(self):
        """Test describe resource group request."""
        req = Prepare.describe_resource_group("test_rg")
        assert req.resource_group == "test_rg"


class TestRegisterLinkRequest:
    """Tests for register_link_request."""

    def test_register_link(self):
        """Test register link request."""
        req = Prepare.register_link_request()
        assert req is not None
