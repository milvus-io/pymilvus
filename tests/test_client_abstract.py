# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

"""Comprehensive tests for pymilvus/client/abstract.py"""

from unittest.mock import MagicMock

import pytest
from pymilvus.client.abstract import (
    AnnSearchRequest,
    BaseRanker,
    CollectionSchema,
    FieldSchema,
    FunctionSchema,
    LoopBase,
    MutationResult,
    RRFRanker,
    StructArrayFieldSchema,
    WeightedRanker,
)
from pymilvus.client.constants import RANKER_TYPE_RRF, RANKER_TYPE_WEIGHTED
from pymilvus.client.types import DataType, FunctionType
from pymilvus.exceptions import DataTypeNotMatchException


class TestFieldSchema:
    """Tests for FieldSchema class."""

    def _create_mock_raw_field(
        self,
        field_id=1,
        name="test_field",
        is_primary_key=False,
        description="Test field",
        auto_id=False,
        data_type=DataType.INT64,
        is_partition_key=False,
        element_type=DataType.NONE,
        is_clustering_key=False,
        default_value=None,
        is_dynamic=False,
        nullable=False,
        is_function_output=False,
        type_params=None,
        index_params=None,
    ):
        """Create a mock raw field object."""
        mock = MagicMock()
        mock.fieldID = field_id
        mock.name = name
        mock.is_primary_key = is_primary_key
        mock.description = description
        mock.autoID = auto_id
        mock.data_type = data_type
        mock.is_partition_key = is_partition_key
        mock.element_type = element_type
        mock.is_clustering_key = is_clustering_key
        mock.default_value = default_value
        mock.is_dynamic = is_dynamic
        mock.nullable = nullable
        mock.is_function_output = is_function_output
        mock.type_params = type_params or []
        mock.index_params = index_params or []
        return mock

    def test_field_schema_basic_init(self):
        """Test FieldSchema basic initialization."""
        raw = self._create_mock_raw_field()
        field = FieldSchema(raw)

        assert field.field_id == 1
        assert field.name == "test_field"
        assert field.is_primary is False
        assert field.description == "Test field"
        assert field.auto_id is False
        assert field.type == DataType.INT64
        assert field.is_partition_key is False
        assert field.element_type == DataType.NONE
        assert field.is_clustering_key is False
        assert field.is_dynamic is False
        assert field.nullable is False
        assert field.is_function_output is False

    def test_field_schema_primary_key(self):
        """Test FieldSchema with primary key."""
        raw = self._create_mock_raw_field(is_primary_key=True, auto_id=True)
        field = FieldSchema(raw)

        assert field.is_primary is True
        assert field.auto_id is True

    def test_field_schema_with_type_params_dim(self):
        """Test FieldSchema with dim type parameter."""
        dim_param = MagicMock()
        dim_param.key = "dim"
        dim_param.value = "128"

        raw = self._create_mock_raw_field(data_type=DataType.FLOAT_VECTOR, type_params=[dim_param])
        field = FieldSchema(raw)

        assert field.params["dim"] == 128

    def test_field_schema_with_type_params_max_length(self):
        """Test FieldSchema with max_length type parameter for VARCHAR."""
        max_length_param = MagicMock()
        max_length_param.key = "max_length"
        max_length_param.value = "256"

        raw = self._create_mock_raw_field(
            data_type=DataType.VARCHAR, type_params=[max_length_param]
        )
        field = FieldSchema(raw)

        assert field.params["max_length"] == 256

    def test_field_schema_with_type_params_max_capacity(self):
        """Test FieldSchema with max_capacity type parameter for ARRAY."""
        max_capacity_param = MagicMock()
        max_capacity_param.key = "max_capacity"
        max_capacity_param.value = "100"

        raw = self._create_mock_raw_field(
            data_type=DataType.ARRAY, type_params=[max_capacity_param]
        )
        field = FieldSchema(raw)

        assert field.params["max_capacity"] == 100

    def test_field_schema_with_mmap_enabled(self):
        """Test FieldSchema with mmap.enabled type parameter."""
        mmap_param = MagicMock()
        mmap_param.key = "mmap.enabled"
        mmap_param.value = "true"

        raw = self._create_mock_raw_field(type_params=[mmap_param])
        field = FieldSchema(raw)

        assert field.params["mmap_enabled"] is True

    def test_field_schema_with_mmap_disabled(self):
        """Test FieldSchema with mmap.enabled set to false."""
        mmap_param = MagicMock()
        mmap_param.key = "mmap.enabled"
        mmap_param.value = "false"

        raw = self._create_mock_raw_field(type_params=[mmap_param])
        field = FieldSchema(raw)

        assert field.params["mmap_enabled"] is False

    def test_field_schema_with_json_params(self):
        """Test FieldSchema with JSON type params."""
        params_param = MagicMock()
        params_param.key = "params"
        params_param.value = '{"key1": "value1", "key2": 42}'

        raw = self._create_mock_raw_field(type_params=[params_param])
        field = FieldSchema(raw)

        assert field.params["params"] == {"key1": "value1", "key2": 42}

    def test_field_schema_with_index_params(self):
        """Test FieldSchema with index parameters."""
        index_param = MagicMock()
        index_param.key = "index_type"
        index_param.value = "IVF_FLAT"

        raw = self._create_mock_raw_field(index_params=[index_param])
        field = FieldSchema(raw)

        assert len(field.indexes) == 1
        assert field.indexes[0]["index_type"] == "IVF_FLAT"

    def test_field_schema_with_json_index_params(self):
        """Test FieldSchema with JSON index parameters."""
        index_param = MagicMock()
        index_param.key = "params"
        index_param.value = '{"nlist": 1024}'

        raw = self._create_mock_raw_field(index_params=[index_param])
        field = FieldSchema(raw)

        assert field.indexes[0]["params"] == {"nlist": 1024}

    def test_field_schema_default_value_none(self):
        """Test FieldSchema with default_value that has WhichOneof returning None."""
        default_val = MagicMock()
        default_val.WhichOneof.return_value = None

        raw = self._create_mock_raw_field(default_value=default_val)
        field = FieldSchema(raw)

        assert field.default_value is None

    def test_field_schema_default_value_with_data(self):
        """Test FieldSchema with default_value that has data."""
        default_val = MagicMock()
        default_val.WhichOneof.return_value = "int_data"

        raw = self._create_mock_raw_field(default_value=default_val)
        field = FieldSchema(raw)

        assert field.default_value == default_val

    def test_field_schema_dict_basic(self):
        """Test FieldSchema dict method basic output."""
        raw = self._create_mock_raw_field()
        field = FieldSchema(raw)
        d = field.dict()

        assert d["field_id"] == 1
        assert d["name"] == "test_field"
        assert d["description"] == "Test field"
        assert d["type"] == DataType.INT64
        assert d["params"] == {}

    def test_field_schema_dict_with_all_flags(self):
        """Test FieldSchema dict method with all optional flags."""
        default_val = MagicMock()
        default_val.WhichOneof.return_value = "int_data"

        raw = self._create_mock_raw_field(
            is_primary_key=True,
            auto_id=True,
            is_partition_key=True,
            is_clustering_key=True,
            is_dynamic=True,
            nullable=True,
            is_function_output=True,
            element_type=DataType.INT32,
            default_value=default_val,
        )
        field = FieldSchema(raw)
        d = field.dict()

        assert d["is_primary"] is True
        assert d["auto_id"] is True
        assert d["is_partition_key"] is True
        assert d["is_clustering_key"] is True
        assert d["is_dynamic"] is True
        assert d["nullable"] is True
        assert d["is_function_output"] is True
        assert d["element_type"] == DataType.INT32
        assert "default_value" in d


class TestStructArrayFieldSchema:
    """Tests for StructArrayFieldSchema class."""

    def _create_mock_raw_struct_field(
        self,
        name="struct_field",
        field_id=1,
        description="Test struct",
        fields=None,
        type_params=None,
    ):
        """Create a mock raw struct array field object."""
        mock = MagicMock()
        mock.name = name
        mock.fieldID = field_id
        mock.description = description
        mock.fields = fields or []
        mock.type_params = type_params or []
        return mock

    def test_struct_array_field_basic_init(self):
        """Test StructArrayFieldSchema basic initialization."""
        raw = self._create_mock_raw_struct_field()
        field = StructArrayFieldSchema(raw)

        assert field.name == "struct_field"
        assert field.field_id == 1
        assert field.description == "Test struct"
        assert field.fields == []
        assert field.params == {}

    def test_struct_array_field_with_nested_fields(self):
        """Test StructArrayFieldSchema with nested fields."""
        # Create a mock nested field
        nested_raw = MagicMock()
        nested_raw.fieldID = 2
        nested_raw.name = "nested_int"
        nested_raw.is_primary_key = False
        nested_raw.description = "Nested integer field"
        nested_raw.autoID = False
        nested_raw.data_type = DataType.INT32
        nested_raw.is_partition_key = False
        nested_raw.element_type = DataType.NONE
        nested_raw.is_clustering_key = False
        nested_raw.default_value = None
        nested_raw.is_dynamic = False
        nested_raw.nullable = False
        nested_raw.is_function_output = False
        nested_raw.type_params = []
        nested_raw.index_params = []

        raw = self._create_mock_raw_struct_field(fields=[nested_raw])
        field = StructArrayFieldSchema(raw)

        assert len(field.fields) == 1
        assert field.fields[0].name == "nested_int"

    def test_struct_array_field_with_type_params(self):
        """Test StructArrayFieldSchema with type parameters."""
        type_param = MagicMock()
        type_param.key = "max_capacity"
        type_param.value = "50"

        raw = self._create_mock_raw_struct_field(type_params=[type_param])
        field = StructArrayFieldSchema(raw)

        assert field.params["max_capacity"] == 50

    def test_struct_array_field_with_mmap_param(self):
        """Test StructArrayFieldSchema with mmap.enabled parameter."""
        mmap_param = MagicMock()
        mmap_param.key = "mmap.enabled"
        mmap_param.value = "true"

        raw = self._create_mock_raw_struct_field(type_params=[mmap_param])
        field = StructArrayFieldSchema(raw)

        assert field.params["mmap_enabled"] is True

    def test_struct_array_field_dict(self):
        """Test StructArrayFieldSchema dict method."""
        type_param = MagicMock()
        type_param.key = "setting"
        type_param.value = "value"

        raw = self._create_mock_raw_struct_field(type_params=[type_param])
        field = StructArrayFieldSchema(raw)
        d = field.dict()

        assert d["field_id"] == 1
        assert d["name"] == "struct_field"
        assert d["description"] == "Test struct"
        assert d["type"] == DataType._ARRAY_OF_STRUCT
        assert d["fields"] == []
        assert d["params"]["setting"] == "value"


class TestFunctionSchema:
    """Tests for FunctionSchema class."""

    def _create_mock_raw_function(
        self,
        name="test_function",
        description="Test function",
        func_id=1,
        func_type=FunctionType.BM25,
        params=None,
        input_field_names=None,
        input_field_ids=None,
        output_field_names=None,
        output_field_ids=None,
    ):
        """Create a mock raw function object."""
        mock = MagicMock()
        mock.name = name
        mock.description = description
        mock.id = func_id
        mock.type = func_type
        mock.params = params or []
        mock.input_field_names = input_field_names or []
        mock.input_field_ids = input_field_ids or []
        mock.output_field_names = output_field_names or []
        mock.output_field_ids = output_field_ids or []
        return mock

    def test_function_schema_basic_init(self):
        """Test FunctionSchema basic initialization."""
        raw = self._create_mock_raw_function()
        func = FunctionSchema(raw)

        assert func.name == "test_function"
        assert func.description == "Test function"
        assert func.id == 1
        assert func.type == FunctionType.BM25
        assert func.params == {}

    def test_function_schema_with_params(self):
        """Test FunctionSchema with parameters."""
        param1 = MagicMock()
        param1.key = "k1"
        param1.value = "v1"
        param2 = MagicMock()
        param2.key = "k2"
        param2.value = "v2"

        raw = self._create_mock_raw_function(params=[param1, param2])
        func = FunctionSchema(raw)

        assert func.params == {"k1": "v1", "k2": "v2"}

    def test_function_schema_with_field_names(self):
        """Test FunctionSchema with input/output field names."""
        raw = self._create_mock_raw_function(
            input_field_names=["text_field"],
            input_field_ids=[1],
            output_field_names=["sparse_vector"],
            output_field_ids=[2],
        )
        func = FunctionSchema(raw)

        assert func.input_field_names == ["text_field"]
        assert func.input_field_ids == [1]
        assert func.output_field_names == ["sparse_vector"]
        assert func.output_field_ids == [2]

    def test_function_schema_dict(self):
        """Test FunctionSchema dict method."""
        raw = self._create_mock_raw_function(
            input_field_names=["input"],
            input_field_ids=[1],
            output_field_names=["output"],
            output_field_ids=[2],
        )
        func = FunctionSchema(raw)
        d = func.dict()

        assert d["name"] == "test_function"
        assert d["id"] == 1
        assert d["description"] == "Test function"
        assert d["type"] == FunctionType.BM25
        assert d["params"] == {}
        assert d["input_field_names"] == ["input"]
        assert d["input_field_ids"] == [1]
        assert d["output_field_names"] == ["output"]
        assert d["output_field_ids"] == [2]


class TestCollectionSchema:
    """Tests for CollectionSchema class."""

    def test_collection_schema_none_raw(self):
        """Test CollectionSchema with None raw data."""
        schema = CollectionSchema(None)

        assert schema.collection_name is None
        assert schema.dict() == {}

    def test_collection_schema_str_none_raw(self):
        """Test CollectionSchema __str__ with None raw data."""
        schema = CollectionSchema(None)
        assert str(schema) == "{}"

    def test_collection_schema_basic_init(self):
        """Test CollectionSchema basic initialization."""
        # Create mock schema
        mock_schema = MagicMock()
        mock_schema.name = "test_collection"
        mock_schema.description = "Test collection"
        mock_schema.enable_dynamic_field = True
        mock_schema.enable_namespace = False
        mock_schema.fields = []
        mock_schema.struct_array_fields = []
        mock_schema.functions = []

        # Create mock raw
        raw = MagicMock()
        raw.schema = mock_schema
        raw.aliases = ["alias1", "alias2"]
        raw.collectionID = 12345
        raw.shards_num = 2
        raw.num_partitions = 4
        raw.created_timestamp = 1704067200
        raw.update_timestamp = 1704153600
        raw.consistency_level = 2  # Bounded
        raw.properties = []

        schema = CollectionSchema(raw)

        assert schema.collection_name == "test_collection"
        assert schema.description == "Test collection"
        assert schema.aliases == ["alias1", "alias2"]
        assert schema.collection_id == 12345
        assert schema.num_shards == 2
        assert schema.num_partitions == 4
        assert schema.enable_dynamic_field is True
        assert schema.enable_namespace is False
        assert schema.created_timestamp == 1704067200
        assert schema.update_timestamp == 1704153600

    def test_collection_schema_with_properties(self):
        """Test CollectionSchema with properties."""
        # Create mock schema
        mock_schema = MagicMock()
        mock_schema.name = "test_collection"
        mock_schema.description = ""
        mock_schema.enable_dynamic_field = False
        mock_schema.enable_namespace = False
        mock_schema.fields = []
        mock_schema.struct_array_fields = []
        mock_schema.functions = []

        # Create mock property
        prop = MagicMock()
        prop.key = "ttl"
        prop.value = "3600"

        # Create mock raw
        raw = MagicMock()
        raw.schema = mock_schema
        raw.aliases = []
        raw.collectionID = 1
        raw.shards_num = 1
        raw.num_partitions = 1
        raw.created_timestamp = 0
        raw.update_timestamp = 0
        raw.consistency_level = 2
        raw.properties = [prop]

        schema = CollectionSchema(raw)

        assert schema.properties == {"ttl": "3600"}

    def test_collection_schema_dict_with_timestamps(self):
        """Test CollectionSchema dict method with timestamps."""
        # Create mock schema
        mock_schema = MagicMock()
        mock_schema.name = "test_collection"
        mock_schema.description = "Test"
        mock_schema.enable_dynamic_field = False
        mock_schema.enable_namespace = False
        mock_schema.fields = []
        mock_schema.struct_array_fields = []
        mock_schema.functions = []

        # Create mock raw
        raw = MagicMock()
        raw.schema = mock_schema
        raw.aliases = []
        raw.collectionID = 1
        raw.shards_num = 1
        raw.num_partitions = 1
        raw.created_timestamp = 1704067200
        raw.update_timestamp = 1704153600
        raw.consistency_level = 2
        raw.properties = []

        schema = CollectionSchema(raw)
        d = schema.dict()

        assert d["created_timestamp"] == 1704067200
        assert d["update_timestamp"] == 1704153600

    def test_collection_schema_dict_without_timestamps(self):
        """Test CollectionSchema dict method without timestamps."""
        # Create mock schema
        mock_schema = MagicMock()
        mock_schema.name = "test_collection"
        mock_schema.description = "Test"
        mock_schema.enable_dynamic_field = False
        mock_schema.enable_namespace = False
        mock_schema.fields = []
        mock_schema.struct_array_fields = []
        mock_schema.functions = []

        # Create mock raw
        raw = MagicMock()
        raw.schema = mock_schema
        raw.aliases = []
        raw.collectionID = 1
        raw.shards_num = 1
        raw.num_partitions = 1
        raw.created_timestamp = 0
        raw.update_timestamp = 0
        raw.consistency_level = 2
        raw.properties = []

        schema = CollectionSchema(raw)
        d = schema.dict()

        assert "created_timestamp" not in d
        assert "update_timestamp" not in d

    def test_collection_schema_rewrite_schema_dict(self):
        """Test CollectionSchema._rewrite_schema_dict method."""
        schema_dict = {
            "fields": [
                {"name": "id", "auto_id": True},
                {"name": "vector", "auto_id": None},
            ]
        }
        CollectionSchema._rewrite_schema_dict(schema_dict)
        assert schema_dict["auto_id"] is True

    def test_collection_schema_rewrite_schema_dict_empty_fields(self):
        """Test CollectionSchema._rewrite_schema_dict with empty fields."""
        schema_dict = {"fields": []}
        CollectionSchema._rewrite_schema_dict(schema_dict)
        assert "auto_id" not in schema_dict


class TestMutationResult:
    """Tests for MutationResult class."""

    def _create_mock_raw_result(
        self,
        int_ids=None,
        str_ids=None,
        insert_cnt=0,
        delete_cnt=0,
        upsert_cnt=0,
        timestamp=0,
        succ_index=None,
        err_index=None,
        cost="0",
    ):
        """Create a mock raw mutation result."""
        mock = MagicMock()

        # Configure IDs
        mock.IDs = MagicMock()
        if int_ids is not None:
            mock.IDs.WhichOneof.return_value = "int_id"
            mock.IDs.int_id.data = int_ids
        elif str_ids is not None:
            mock.IDs.WhichOneof.return_value = "str_id"
            mock.IDs.str_id.data = str_ids
        else:
            mock.IDs.WhichOneof.return_value = None

        mock.insert_cnt = insert_cnt
        mock.delete_cnt = delete_cnt
        mock.upsert_cnt = upsert_cnt
        mock.timestamp = timestamp
        mock.succ_index = succ_index or []
        mock.err_index = err_index or []

        # Configure status with extra_info
        mock.status = MagicMock()
        mock.status.extra_info = {"report_value": cost}

        return mock

    def test_mutation_result_with_int_ids(self):
        """Test MutationResult with integer primary keys."""
        raw = self._create_mock_raw_result(
            int_ids=[1, 2, 3], insert_cnt=3, timestamp=12345, succ_index=[0, 1, 2]
        )
        result = MutationResult(raw)

        assert result.primary_keys == [1, 2, 3]
        assert result.insert_count == 3
        assert result.timestamp == 12345
        assert result.succ_count == 3
        assert result.succ_index == [0, 1, 2]

    def test_mutation_result_with_str_ids(self):
        """Test MutationResult with string primary keys."""
        raw = self._create_mock_raw_result(str_ids=["a", "b", "c"], insert_cnt=3, timestamp=12345)
        result = MutationResult(raw)

        assert result.primary_keys == ["a", "b", "c"]

    def test_mutation_result_delete(self):
        """Test MutationResult for delete operation."""
        raw = self._create_mock_raw_result(delete_cnt=5)
        result = MutationResult(raw)

        assert result.delete_count == 5

    def test_mutation_result_upsert(self):
        """Test MutationResult for upsert operation."""
        raw = self._create_mock_raw_result(upsert_cnt=10)
        result = MutationResult(raw)

        assert result.upsert_count == 10

    def test_mutation_result_with_errors(self):
        """Test MutationResult with error indices."""
        raw = self._create_mock_raw_result(
            int_ids=[1, 2, 3], insert_cnt=2, succ_index=[0, 2], err_index=[1]
        )
        result = MutationResult(raw)

        assert result.succ_count == 2
        assert result.err_count == 1
        assert result.err_index == [1]

    def test_mutation_result_cost(self):
        """Test MutationResult cost property."""
        raw = self._create_mock_raw_result(int_ids=[1], insert_cnt=1, cost="42")
        result = MutationResult(raw)

        assert result.cost == 42

    def test_mutation_result_str_with_cost(self):
        """Test MutationResult __str__ with cost."""
        raw = self._create_mock_raw_result(
            int_ids=[1], insert_cnt=1, delete_cnt=2, upsert_cnt=3, timestamp=100, cost="50"
        )
        result = MutationResult(raw)
        s = str(result)

        assert "insert count: 1" in s
        assert "delete count: 2" in s
        assert "upsert count: 3" in s
        assert "timestamp: 100" in s
        assert "cost: 50" in s

    def test_mutation_result_str_without_cost(self):
        """Test MutationResult __str__ without cost."""
        raw = self._create_mock_raw_result(int_ids=[1], insert_cnt=1, cost="0")
        result = MutationResult(raw)
        s = str(result)

        assert "cost" not in s

    def test_mutation_result_repr(self):
        """Test MutationResult __repr__ is same as __str__."""
        raw = self._create_mock_raw_result(int_ids=[1], insert_cnt=1)
        result = MutationResult(raw)

        assert str(result) == repr(result)

    def test_mutation_result_no_extra_info(self):
        """Test MutationResult when status has no extra_info."""
        raw = self._create_mock_raw_result(int_ids=[1], insert_cnt=1)
        raw.status.extra_info = None
        result = MutationResult(raw)

        assert result.cost == 0


class TestBaseRanker:
    """Tests for BaseRanker class."""

    def test_base_ranker_init(self):
        """Test BaseRanker initialization."""
        ranker = BaseRanker()
        # __int__ method exists but returns None
        assert ranker.__int__() is None

    def test_base_ranker_dict(self):
        """Test BaseRanker dict method."""
        ranker = BaseRanker()
        assert ranker.dict() == {}

    def test_base_ranker_str(self):
        """Test BaseRanker __str__ method."""
        ranker = BaseRanker()
        assert str(ranker) == "{}"


class TestRRFRanker:
    """Tests for RRFRanker class."""

    def test_rrf_ranker_default_k(self):
        """Test RRFRanker with default k value."""
        ranker = RRFRanker()

        assert ranker._strategy == RANKER_TYPE_RRF
        assert ranker._k == 60

    def test_rrf_ranker_custom_k(self):
        """Test RRFRanker with custom k value."""
        ranker = RRFRanker(k=100)

        assert ranker._k == 100

    def test_rrf_ranker_dict(self):
        """Test RRFRanker dict method."""
        ranker = RRFRanker(k=50)
        d = ranker.dict()

        assert d["strategy"] == RANKER_TYPE_RRF
        assert d["params"]["k"] == 50

    def test_rrf_ranker_str(self):
        """Test RRFRanker __str__ method."""
        ranker = RRFRanker(k=60)
        s = str(ranker)

        assert "rrf" in s
        assert "60" in s


class TestWeightedRanker:
    """Tests for WeightedRanker class."""

    def test_weighted_ranker_basic(self):
        """Test WeightedRanker basic initialization."""
        ranker = WeightedRanker(0.5, 0.3, 0.2)

        assert ranker._strategy == RANKER_TYPE_WEIGHTED
        assert ranker._weights == [0.5, 0.3, 0.2]
        assert ranker._norm_score is True

    def test_weighted_ranker_no_norm_score(self):
        """Test WeightedRanker with norm_score=False."""
        ranker = WeightedRanker(0.5, 0.5, norm_score=False)

        assert ranker._norm_score is False

    def test_weighted_ranker_int_weights(self):
        """Test WeightedRanker with integer weights."""
        ranker = WeightedRanker(1, 2, 3)

        assert ranker._weights == [1, 2, 3]

    def test_weighted_ranker_invalid_bool_weight(self):
        """Test WeightedRanker raises error for bool weight."""
        with pytest.raises(TypeError) as exc_info:
            WeightedRanker(0.5, True, 0.5)

        assert "Weight must be a number" in str(exc_info.value)

    def test_weighted_ranker_invalid_string_weight(self):
        """Test WeightedRanker raises error for string weight."""
        with pytest.raises(TypeError) as exc_info:
            WeightedRanker(0.5, "0.5", 0.5)

        assert "Weight must be a number" in str(exc_info.value)

    def test_weighted_ranker_invalid_none_weight(self):
        """Test WeightedRanker raises error for None weight."""
        with pytest.raises(TypeError) as exc_info:
            WeightedRanker(0.5, None)

        assert "Weight must be a number" in str(exc_info.value)

    def test_weighted_ranker_dict(self):
        """Test WeightedRanker dict method."""
        ranker = WeightedRanker(0.6, 0.4, norm_score=True)
        d = ranker.dict()

        assert d["strategy"] == RANKER_TYPE_WEIGHTED
        assert d["params"]["weights"] == [0.6, 0.4]
        assert d["params"]["norm_score"] is True

    def test_weighted_ranker_str(self):
        """Test WeightedRanker __str__ method."""
        ranker = WeightedRanker(0.5, 0.5)
        s = str(ranker)

        assert "weighted" in s
        assert "0.5" in s


class TestAnnSearchRequest:
    """Tests for AnnSearchRequest class."""

    def test_ann_search_request_basic(self):
        """Test AnnSearchRequest basic initialization."""
        data = [[0.1, 0.2, 0.3, 0.4]]
        request = AnnSearchRequest(
            data=data,
            anns_field="vector_field",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        assert request.data == data
        assert request.anns_field == "vector_field"
        assert request.param == {"metric_type": "L2", "params": {"nprobe": 10}}
        assert request.limit == 10
        assert request.expr is None
        assert request.expr_params is None

    def test_ann_search_request_with_expr(self):
        """Test AnnSearchRequest with filter expression."""
        data = [[0.1, 0.2, 0.3, 0.4]]
        request = AnnSearchRequest(
            data=data,
            anns_field="vector_field",
            param={"metric_type": "L2"},
            limit=10,
            expr="id > 100",
        )

        assert request.expr == "id > 100"

    def test_ann_search_request_with_expr_params(self):
        """Test AnnSearchRequest with expression parameters."""
        data = [[0.1, 0.2, 0.3, 0.4]]
        request = AnnSearchRequest(
            data=data,
            anns_field="vector_field",
            param={"metric_type": "L2"},
            limit=10,
            expr="id > {min_id}",
            expr_params={"min_id": 100},
        )

        assert request.expr_params == {"min_id": 100}

    def test_ann_search_request_invalid_expr_type(self):
        """Test AnnSearchRequest raises error for invalid expr type."""
        with pytest.raises(DataTypeNotMatchException):
            AnnSearchRequest(
                data=[[0.1, 0.2]],
                anns_field="vector_field",
                param={},
                limit=10,
                expr=123,  # Invalid type
            )

    def test_ann_search_request_str(self):
        """Test AnnSearchRequest __str__ method."""
        data = [[0.1, 0.2, 0.3, 0.4]]
        request = AnnSearchRequest(
            data=data,
            anns_field="vector_field",
            param={"metric_type": "L2"},
            limit=10,
            expr="id > 0",
        )
        s = str(request)

        assert "vector_field" in s
        assert "10" in s
        assert "id > 0" in s


class TestLoopBase:
    """Tests for LoopBase abstract class."""

    class ConcreteLoop(LoopBase):
        """Concrete implementation for testing."""

        def __init__(self, items):
            super().__init__()
            self._items = items

        def __len__(self):
            return len(self._items)

        def get__item(self, item):
            return self._items[item]

    def test_loop_base_iteration(self):
        """Test LoopBase iteration."""
        loop = self.ConcreteLoop([1, 2, 3])
        result = list(loop)

        assert result == [1, 2, 3]

    def test_loop_base_getitem_single(self):
        """Test LoopBase single item access."""
        loop = self.ConcreteLoop([1, 2, 3])

        assert loop[0] == 1
        assert loop[1] == 2
        assert loop[2] == 3

    def test_loop_base_getitem_index_error(self):
        """Test LoopBase index out of range."""
        loop = self.ConcreteLoop([1, 2, 3])

        with pytest.raises(IndexError) as exc_info:
            _ = loop[5]

        assert "Index out of range" in str(exc_info.value)

    def test_loop_base_getitem_slice(self):
        """Test LoopBase slice access."""
        loop = self.ConcreteLoop([1, 2, 3, 4, 5])

        assert loop[1:4] == [2, 3, 4]
        assert loop[:3] == [1, 2, 3]
        assert loop[2:] == [3, 4, 5]
        assert loop[::2] == [1, 3, 5]

    def test_loop_base_getitem_slice_with_none(self):
        """Test LoopBase slice with None values."""
        loop = self.ConcreteLoop([1, 2, 3])

        assert loop[:] == [1, 2, 3]
        assert loop[None:None:None] == [1, 2, 3]

    def test_loop_base_str(self):
        """Test LoopBase __str__ method."""
        loop = self.ConcreteLoop([1, 2, 3])
        s = str(loop)

        assert "1" in s
        assert "2" in s
        assert "3" in s

    def test_loop_base_str_more_than_10(self):
        """Test LoopBase __str__ with more than 10 items."""
        loop = self.ConcreteLoop(list(range(15)))
        s = str(loop)

        # Should only show first 10 items
        assert "9" in s

    def test_loop_base_iter_reset(self):
        """Test LoopBase iterator reset after StopIteration."""
        loop = self.ConcreteLoop([1, 2])

        # First iteration
        result1 = list(loop)
        assert result1 == [1, 2]

        # Second iteration should work again
        result2 = list(loop)
        assert result2 == [1, 2]

    def test_loop_base_abstract_get_item(self):
        """Test LoopBase.get__item raises NotImplementedError."""
        base = LoopBase()

        with pytest.raises(NotImplementedError):
            base.get__item(0)
