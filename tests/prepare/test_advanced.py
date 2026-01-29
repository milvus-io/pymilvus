"""Tests for advanced Prepare methods (bulk operations, replication, analyzers)."""

import pytest
from pymilvus import Function, FunctionType
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError


class TestBulkInsertRequests:
    """Tests for bulk insert related requests."""

    def test_do_bulk_insert_basic(self):
        """Test basic bulk insert request."""
        req = Prepare.do_bulk_insert(
            collection_name="test_coll",
            partition_name="part1",
            files=["file1.json", "file2.json"],
        )
        assert req.collection_name == "test_coll"
        assert req.partition_name == "part1"
        assert list(req.files) == ["file1.json", "file2.json"]

    def test_do_bulk_insert_with_channel_names(self):
        """Test bulk insert with channel names."""
        req = Prepare.do_bulk_insert(
            collection_name="test_coll",
            partition_name="",
            files=["file1.json"],
            channel_names=["channel1", "channel2"],
        )
        assert list(req.channel_names) == ["channel1", "channel2"]

    @pytest.mark.parametrize(
        "option_key,value",
        [
            pytest.param("bucket", "my_bucket", id="bucket"),
            pytest.param("backup", "backup_path", id="backup"),
            pytest.param("sep", ",", id="sep"),
            pytest.param("nullkey", "NULL", id="nullkey"),
        ],
    )
    def test_do_bulk_insert_with_options(self, option_key, value):
        """Test bulk insert with various options."""
        kwargs = {option_key: value}
        req = Prepare.do_bulk_insert(
            collection_name="test_coll",
            partition_name="",
            files=["file1.json"],
            **kwargs,
        )
        option_keys = [opt.key for opt in req.options]
        assert option_key in option_keys

    def test_get_bulk_insert_state(self):
        """Test get bulk insert state."""
        req = Prepare.get_bulk_insert_state(12345)
        assert req.task == 12345

    @pytest.mark.parametrize(
        "task_id",
        [
            pytest.param(None, id="none"),
            pytest.param("12345", id="string"),
            pytest.param(12.5, id="float"),
        ],
    )
    def test_get_bulk_insert_state_invalid_id(self, task_id):
        """Test get bulk insert state with invalid task ID."""
        with pytest.raises(ParamError, match="not an integer"):
            Prepare.get_bulk_insert_state(task_id)

    def test_list_bulk_insert_tasks(self):
        """Test list bulk insert tasks."""
        req = Prepare.list_bulk_insert_tasks(limit=100, collection_name="test_coll")
        assert req.collection_name == "test_coll"
        assert req.limit == 100

    @pytest.mark.parametrize(
        "limit",
        [
            pytest.param(None, id="none"),
            pytest.param("100", id="string"),
        ],
    )
    def test_list_bulk_insert_tasks_invalid_limit(self, limit):
        """Test list bulk insert tasks with invalid limit."""
        with pytest.raises(ParamError, match="not an integer"):
            Prepare.list_bulk_insert_tasks(limit=limit, collection_name="test_coll")


class TestRunAnalyzer:
    """Tests for run_analyzer."""

    @pytest.mark.parametrize(
        "texts,expected_count",
        [
            pytest.param("hello world", 1, id="single_string"),
            pytest.param(["hello", "world"], 2, id="list_of_strings"),
            pytest.param(["one", "two", "three"], 3, id="three_strings"),
        ],
    )
    def test_analyzer_text_input(self, texts, expected_count):
        """Test analyzer with various text inputs."""
        req = Prepare.run_analyzer(texts=texts)
        assert len(req.placeholder) == expected_count

    @pytest.mark.parametrize(
        "analyzer_params,expected",
        [
            pytest.param({"type": "standard"}, '{"type":"standard"}', id="dict_params"),
            pytest.param('{"type":"standard"}', '{"type":"standard"}', id="string_params"),
            pytest.param({"k": "v", "n": 1}, '{"k":"v","n":1}', id="dict_multi_params"),
        ],
    )
    def test_analyzer_params(self, analyzer_params, expected):
        """Test analyzer with various parameter formats."""
        req = Prepare.run_analyzer(texts="hello", analyzer_params=analyzer_params)
        assert req.analyzer_params == expected

    @pytest.mark.parametrize(
        "option_key,option_value",
        [
            pytest.param("with_hash", True, id="with_hash_true"),
            pytest.param("with_hash", False, id="with_hash_false"),
            pytest.param("with_detail", True, id="with_detail_true"),
            pytest.param("with_detail", False, id="with_detail_false"),
        ],
    )
    def test_analyzer_boolean_options(self, option_key, option_value):
        """Test analyzer with boolean options."""
        kwargs = {option_key: option_value}
        req = Prepare.run_analyzer(texts="hello", **kwargs)
        assert getattr(req, option_key) is option_value

    @pytest.mark.parametrize(
        "collection_name,field_name",
        [
            pytest.param("test_coll", "text_field", id="both_specified"),
            pytest.param("coll", "field", id="short_names"),
            pytest.param("my_collection", "content", id="descriptive_names"),
        ],
    )
    def test_analyzer_collection_and_field(self, collection_name, field_name):
        """Test analyzer with collection and field names."""
        req = Prepare.run_analyzer(
            texts="hello",
            collection_name=collection_name,
            field_name=field_name,
        )
        assert req.collection_name == collection_name
        assert req.field_name == field_name

    @pytest.mark.parametrize(
        "analyzer_names,expected",
        [
            pytest.param("standard", ["standard"], id="single_string"),
            pytest.param(["standard"], ["standard"], id="single_list"),
            pytest.param(["standard", "whitespace"], ["standard", "whitespace"], id="multiple"),
            pytest.param(["a", "b", "c"], ["a", "b", "c"], id="three_analyzers"),
        ],
    )
    def test_analyzer_names(self, analyzer_names, expected):
        """Test analyzer with various analyzer name formats."""
        req = Prepare.run_analyzer(texts="hello", analyzer_names=analyzer_names)
        assert list(req.analyzer_names) == expected


class TestUpdateReplicateConfiguration:
    """Tests for update_replicate_configuration_request."""

    def test_replicate_config_neither_provided(self):
        """Test replicate config without clusters or topology."""
        with pytest.raises(ParamError, match="must be provided"):
            Prepare.update_replicate_configuration_request()

    @pytest.mark.parametrize(
        "cluster,error_match",
        [
            pytest.param(
                {"connection_param": {"uri": "localhost:19530"}},
                "cluster_id is required",
                id="missing_cluster_id",
            ),
            pytest.param(
                {"cluster_id": "cluster1"},
                "connection_param is required",
                id="missing_connection_param",
            ),
            pytest.param(
                {"cluster_id": "cluster1", "connection_param": {"token": "token123"}},
                "uri is required",
                id="missing_uri",
            ),
        ],
    )
    def test_replicate_config_cluster_validation(self, cluster, error_match):
        """Test replicate config cluster validation errors."""
        with pytest.raises(ParamError, match=error_match):
            Prepare.update_replicate_configuration_request(clusters=[cluster])

    @pytest.mark.parametrize(
        "clusters,expected_count",
        [
            pytest.param(
                [{"cluster_id": "c1", "connection_param": {"uri": "localhost:19530"}}],
                1,
                id="single_cluster",
            ),
            pytest.param(
                [
                    {"cluster_id": "c1", "connection_param": {"uri": "localhost:19530"}},
                    {"cluster_id": "c2", "connection_param": {"uri": "localhost:19531"}},
                ],
                2,
                id="two_clusters",
            ),
        ],
    )
    def test_replicate_config_with_clusters(self, clusters, expected_count):
        """Test replicate config with valid clusters."""
        req = Prepare.update_replicate_configuration_request(clusters=clusters)
        assert len(req.replicate_configuration.clusters) == expected_count

    @pytest.mark.parametrize(
        "extra_field,extra_value,attr_check",
        [
            pytest.param("token", "mytoken", lambda c: c.connection_param.token, id="with_token"),
            pytest.param(
                "pchannels",
                ["ch1", "ch2"],
                lambda c: list(c.pchannels),
                id="with_pchannels",
            ),
        ],
    )
    def test_replicate_config_cluster_optional_fields(self, extra_field, extra_value, attr_check):
        """Test replicate config with optional cluster fields."""
        cluster = {
            "cluster_id": "cluster1",
            "connection_param": {"uri": "localhost:19530"},
        }
        if extra_field == "token":
            cluster["connection_param"]["token"] = extra_value
        else:
            cluster[extra_field] = extra_value

        req = Prepare.update_replicate_configuration_request(clusters=[cluster])
        result = attr_check(req.replicate_configuration.clusters[0])
        if isinstance(extra_value, list):
            assert result == extra_value
        else:
            assert result == extra_value

    @pytest.mark.parametrize(
        "topology,error_match",
        [
            pytest.param(
                {"target_cluster_id": "cluster2"},
                "source_cluster_id is required",
                id="missing_source",
            ),
            pytest.param(
                {"source_cluster_id": "cluster1"},
                "target_cluster_id is required",
                id="missing_target",
            ),
        ],
    )
    def test_replicate_config_topology_validation(self, topology, error_match):
        """Test replicate config topology validation errors."""
        with pytest.raises(ParamError, match=error_match):
            Prepare.update_replicate_configuration_request(cross_cluster_topology=[topology])

    def test_replicate_config_with_topology(self):
        """Test replicate config with cross cluster topology."""
        topology = [{"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}]
        req = Prepare.update_replicate_configuration_request(cross_cluster_topology=topology)
        assert len(req.replicate_configuration.cross_cluster_topology) == 1

    def test_replicate_config_with_both(self):
        """Test replicate config with both clusters and topology."""
        clusters = [
            {"cluster_id": "cluster1", "connection_param": {"uri": "localhost:19530"}},
            {"cluster_id": "cluster2", "connection_param": {"uri": "localhost:19531"}},
        ]
        topology = [{"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}]
        req = Prepare.update_replicate_configuration_request(
            clusters=clusters,
            cross_cluster_topology=topology,
        )
        assert len(req.replicate_configuration.clusters) == 2
        assert len(req.replicate_configuration.cross_cluster_topology) == 1


class TestConvertFunctionToFunctionSchema:
    """Tests for convert_function_to_function_schema."""

    @pytest.mark.parametrize(
        "name,func_type,input_fields,output_fields",
        [
            pytest.param(
                "test_func",
                FunctionType.TEXTEMBEDDING,
                ["text"],
                ["embedding"],
                id="text_embedding",
            ),
            pytest.param(
                "rerank_func",
                FunctionType.RERANK,
                ["query", "doc"],
                ["score"],
                id="rerank",
            ),
            pytest.param(
                "multi_input",
                FunctionType.TEXTEMBEDDING,
                ["title", "body"],
                ["vec1", "vec2"],
                id="multiple_fields",
            ),
        ],
    )
    def test_convert_function_basic(self, name, func_type, input_fields, output_fields):
        """Test converting basic functions to schema."""
        func = Function(
            name,
            func_type,
            input_field_names=input_fields,
            output_field_names=output_fields,
        )
        result = Prepare.convert_function_to_function_schema(func)
        assert result.name == name
        assert result.type == func_type
        assert list(result.input_field_names) == input_fields
        assert list(result.output_field_names) == output_fields

    @pytest.mark.parametrize(
        "params,expected_keys",
        [
            pytest.param({"model": "test_model"}, ["model"], id="single_param"),
            pytest.param({"model": "test", "dim": 128}, ["model", "dim"], id="multiple_params"),
            pytest.param({"k": 1.5, "b": 0.75}, ["k", "b"], id="float_params"),
        ],
    )
    def test_convert_function_with_params(self, params, expected_keys):
        """Test converting function with various parameters."""
        func = Function(
            "test_func",
            FunctionType.TEXTEMBEDDING,
            input_field_names=["text"],
            output_field_names=["embedding"],
            params=params,
        )
        result = Prepare.convert_function_to_function_schema(func)
        param_dict = {p.key: p.value for p in result.params}
        for key in expected_keys:
            assert key in param_dict

    @pytest.mark.parametrize(
        "description",
        [
            pytest.param("Test function description", id="simple"),
            pytest.param("A longer description with multiple words", id="longer"),
            pytest.param("", id="empty"),
        ],
    )
    def test_convert_function_with_description(self, description):
        """Test converting function with description."""
        func = Function(
            "test_func",
            FunctionType.TEXTEMBEDDING,
            input_field_names=["text"],
            output_field_names=["embedding"],
            description=description,
        )
        result = Prepare.convert_function_to_function_schema(func)
        assert result.description == description


class TestEmptyMethod:
    """Tests for deprecated empty method."""

    def test_empty_raises_deprecation_warning(self):
        """Test that empty method raises DeprecationWarning."""
        with pytest.raises(DeprecationWarning):
            Prepare.empty()
