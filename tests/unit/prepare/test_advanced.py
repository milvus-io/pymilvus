"""Tests for advanced Prepare methods (bulk operations, replication, analyzers)."""

from typing import ClassVar

import pytest
from pymilvus import Function, FunctionType
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2


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

    def test_replicate_config_clusters_required(self):
        """Test replicate config always requires clusters."""
        with pytest.raises(ParamError, match="must be provided"):
            Prepare.update_replicate_configuration_request()

    def test_replicate_config_clusters_required_even_with_force_promote(self):
        """Test that clusters is required even when force_promote=True."""
        with pytest.raises(ParamError, match="must be provided"):
            Prepare.update_replicate_configuration_request(force_promote=True)

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
        clusters = [{"cluster_id": "c1", "connection_param": {"uri": "localhost:19530"}}]
        with pytest.raises(ParamError, match=error_match):
            Prepare.update_replicate_configuration_request(
                clusters=clusters, cross_cluster_topology=[topology]
            )

    def test_replicate_config_with_topology(self):
        """Test replicate config with cross cluster topology."""
        clusters = [{"cluster_id": "cluster1", "connection_param": {"uri": "localhost:19530"}}]
        topology = [{"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}]
        req = Prepare.update_replicate_configuration_request(
            clusters=clusters, cross_cluster_topology=topology
        )
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

    def test_force_promote_without_clusters_or_topology(self):
        """Test force promote still requires clusters."""
        with pytest.raises(ParamError, match="must be provided"):
            Prepare.update_replicate_configuration_request(force_promote=True)

    def test_force_promote_sets_flag(self):
        """Test that force_promote=True sets the flag in the request."""
        clusters = [{"cluster_id": "c1", "connection_param": {"uri": "localhost:19530"}}]
        req = Prepare.update_replicate_configuration_request(clusters=clusters, force_promote=True)
        assert req.force_promote is True

    def test_force_promote_default_false(self):
        """Test that force_promote defaults to False."""
        clusters = [{"cluster_id": "c1", "connection_param": {"uri": "localhost:19530"}}]
        req = Prepare.update_replicate_configuration_request(clusters=clusters)
        assert req.force_promote is False


class TestGetReplicateInfoRequest:
    """Tests for Prepare.get_replicate_info_request."""

    def test_builds_correct_request(self):
        req = Prepare.get_replicate_info_request(
            source_cluster_id="src",
            target_pchannel="by-dev-rootcoord-dml_0",
        )
        assert req.source_cluster_id == "src"
        assert req.target_pchannel == "by-dev-rootcoord-dml_0"

    def test_missing_source_cluster_id_raises(self):
        with pytest.raises(ParamError, match="source_cluster_id"):
            Prepare.get_replicate_info_request(
                source_cluster_id="",
                target_pchannel="ch0",
            )

    def test_missing_target_pchannel_raises(self):
        with pytest.raises(ParamError, match="target_pchannel"):
            Prepare.get_replicate_info_request(
                source_cluster_id="src",
                target_pchannel="",
            )

    def test_none_source_cluster_id_raises(self):
        with pytest.raises(ParamError, match="source_cluster_id"):
            Prepare.get_replicate_info_request(
                source_cluster_id=None,
                target_pchannel="ch0",
            )

    def test_none_target_pchannel_raises(self):
        with pytest.raises(ParamError, match="target_pchannel"):
            Prepare.get_replicate_info_request(
                source_cluster_id="src",
                target_pchannel=None,
            )


class TestDumpMessagesRequest:
    """Tests for Prepare.dump_messages_request."""

    _MSG_ID: ClassVar[dict] = {"id": "msg-1", "wal_name": "Pulsar"}

    def test_builds_correct_request(self):
        req = Prepare.dump_messages_request(
            pchannel="by-dev-rootcoord-dml_0",
            start_message_id=self._MSG_ID,
            start_timetick=100,
            end_timetick=200,
        )
        assert req.pchannel == "by-dev-rootcoord-dml_0"
        assert req.start_message_id.id == "msg-1"
        assert req.start_message_id.WAL_name == common_pb2.WALName.Pulsar
        assert req.start_timetick == 100
        assert req.end_timetick == 200

    def test_timeticks_default_to_zero(self):
        req = Prepare.dump_messages_request(
            pchannel="ch0",
            start_message_id=self._MSG_ID,
        )
        assert req.start_timetick == 0
        assert req.end_timetick == 0

    def test_missing_pchannel_raises(self):
        with pytest.raises(ParamError, match="pchannel"):
            Prepare.dump_messages_request(pchannel="", start_message_id=self._MSG_ID)

    def test_none_pchannel_raises(self):
        with pytest.raises(ParamError, match="pchannel"):
            Prepare.dump_messages_request(pchannel=None, start_message_id=self._MSG_ID)

    def test_missing_start_message_id_raises(self):
        with pytest.raises(ParamError, match="start_message_id"):
            Prepare.dump_messages_request(pchannel="ch0", start_message_id=None)

    def test_empty_start_message_id_raises(self):
        with pytest.raises(ParamError, match="start_message_id"):
            Prepare.dump_messages_request(pchannel="ch0", start_message_id={})

    def test_non_dict_start_message_id_raises(self):
        with pytest.raises(ParamError, match="start_message_id"):
            Prepare.dump_messages_request(pchannel="ch0", start_message_id="msg-1")

    def test_missing_id_key_raises(self):
        with pytest.raises(ParamError, match=r"start_message_id\.id"):
            Prepare.dump_messages_request(pchannel="ch0", start_message_id={"wal_name": "Pulsar"})

    def test_invalid_wal_name_raises(self):
        with pytest.raises(ParamError, match="wal_name"):
            Prepare.dump_messages_request(
                pchannel="ch0", start_message_id={"id": "m", "wal_name": "NotAWal"}
            )

    def test_missing_wal_name_raises(self):
        with pytest.raises(ParamError, match="wal_name"):
            Prepare.dump_messages_request(pchannel="ch0", start_message_id={"id": "m"})


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
