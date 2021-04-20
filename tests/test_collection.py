import sys
sys.path.append(".")
from unittest import mock

import grpc
import pytest
from grpc._channel import _UnaryUnaryMultiCallable as Uum

from milvus import Milvus, MetricType, IndexType, ParamError

from factorys import collection_schema_factory

from utils import MockGrpcError


class TestCreateCollection:
    def test_create_collection_normal(self, gcon):
        collection_param = collection_schema_factory()
        status = gcon.create_collection(collection_param)
        assert status.OK(), status.message

        status = gcon.drop_collection(collection_param["collection_name"])
        assert status.OK(), status.message

    def test_create_collection_repeat(self, gcon):
        collection_param = collection_schema_factory()
        status = gcon.create_collection(collection_param)
        assert status.OK(), status.message

        status = gcon.create_collection(collection_param)
        assert not status.OK()

        status = gcon.drop_collection(collection_param["collection_name"])
        assert status.OK(), status.message

    @pytest.mark.parametrize("metric",
                             [MetricType.L2, MetricType.IP, MetricType.HAMMING,
                              MetricType.JACCARD, MetricType.TANIMOTO,
                              MetricType.SUPERSTRUCTURE, MetricType.SUBSTRUCTURE])
    def test_create_collection_whole_metric(self, metric, gcon):
        collection_param = {
            "collection_name": "test_create_collection_metric_" + str(metric),
            "dimension": 128,
            "metric_type": metric
        }

        status = gcon.create_collection(collection_param)
        assert status.OK()
        status = gcon.drop_collection(collection_param["collection_name"])
        assert status.OK(), status.message

    def test_create_collection_default_index_file_size(self, gcon):
        collection_param = {
            "collection_name": "test_create_collection_default_index_file_size",
            "dimension": 128,
            "metric_type": MetricType.L2
        }
        status = gcon.create_collection(collection_param)
        assert status.OK(), status.message

        status = gcon.drop_collection(collection_param["collection_name"])
        assert status.OK(), status.message

    def test_create_collection_default_metric_type(self, gcon):
        collection_param = {
            "collection_name": "test_create_collection_default_metric_type",
            "dimension": 128,
            "index_file_size": 100
        }
        status = gcon.create_collection(collection_param)
        assert status.OK(), status.message

        status = gcon.drop_collection(collection_param["collection_name"])
        assert status.OK(), status.message

    @pytest.mark.parametrize("metric", ["123", None, MetricType.INVALID, -1, 1000])
    def test_create_collection_invalid_metric(self, metric, gcon):
        collection_param = {
            "collection_name": "test_create_collection_invalid_metric_" + str(metric),
            "dimension": 128,
            "metric_type": metric
        }
        with pytest.raises(ParamError):
            gcon.create_collection(collection_param)

    @pytest.mark.parametrize("dim", ["123", None, [1]])
    def test_create_collection_invalid_dim(self, dim, gcon):
        collection_param = {
            "collection_name": "test_create_collection_invalid_dim_" + str(dim),
            "dimension": dim,
            "metric_type": MetricType.L2
        }
        with pytest.raises(ParamError):
            gcon.create_collection(collection_param)

    @pytest.mark.parametrize("index_file_size", ["123", None, [1]])
    def test_create_collection_invalid_index_file_size(self, index_file_size, gcon):
        collection_param = {
            "collection_name": "test_create_collection_invalid_index_file_size_" + str(index_file_size),
            "dimension": 128,
            "metric_type": MetricType.L2,
            "index_file_size": index_file_size
        }
        with pytest.raises(ParamError):
            gcon.create_collection(collection_param)

    def test_create_collection_exception(self, gcon):
        collection_param = {
            "collection_name": "test_create_collection_exceptions",
            "dimension": 128,
            "metric_type": MetricType.L2,
            "index_file_size": 10
        }

        mock_grpc_timeout = mock.MagicMock(side_effect=grpc.FutureTimeoutError())
        with mock.patch.object(Uum, 'future', mock_grpc_timeout):
            status = gcon.create_collection(collection_param)
            assert not status.OK()

        mock_grpc_error = mock.MagicMock(side_effect=MockGrpcError())
        with mock.patch.object(Uum, 'future', mock_grpc_error):
            status = gcon.create_collection(collection_param)
            assert not status.OK()

        with pytest.raises(Exception):
            mock_exception = mock.MagicMock(side_effect=Exception("error"))
            with mock.patch.object(Uum, 'future', mock_exception):
                status = gcon.create_collection(collection_param)
                assert not status.OK()


class TestHasCollection:
    def test_has_collection_normal(self, gcon, gcollection):
        status, ok = gcon.has_collection(gcollection)
        assert status.OK()
        assert ok

    @pytest.mark.parametrize("collection", [None, 123, [123], {}, ""])
    def test_has_collection_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.has_collection(collection)

    def test_has_collection_non_existent(self, gcon):
        status, ok = gcon.has_collection("sfsfsfsfsfsfsfsfsfsf")
        assert not (status.OK() and ok)


class TestDescribeCollection:
    def test_get_collection_info_normal(self, gcon, gcollection):
        status, info = gcon.get_collection_info(gcollection)
        assert status.OK()
        assert info.dimension == 128

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_get_collection_info_with_invalid_collection(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.get_collection_info(collection)

    def test_get_collection_info_non_existent(self, gcon):
        status, _ = gcon.get_collection_info("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestCountCollection:
    def test_count_entities_normal(self, gcon, gvector):
        status, count = gcon.count_entities(gvector)
        assert status.OK()
        assert count == 10000

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_count_entities_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.count_entities(collection)

    def test_count_entities_non_existent(self, gcon):
        status, _ = gcon.count_entities("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestCollectinStats:
    def test_get_collection_stats_normal(self, gcon, gvector):
        status, info = gcon.get_collection_stats(gvector)
        assert status.OK()
        assert info["row_count"] == 10000

        par0_stat = info["partitions"][0]
        assert par0_stat["tag"] == "_default"
        assert par0_stat["row_count"] == 10000

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_get_collection_stats_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.get_collection_stats(collection)

    def test_count_entities_non_existent(self, gcon):
        status, _ = gcon.get_collection_stats("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestShowCollections:
    def test_list_collections(self, gcon, gcollection):
        status, names = gcon.list_collections()
        assert status.OK()
        assert gcollection in names


class TestDropCollection:
    def test_drop_collection_normal(self, gcon):
        status = gcon.create_collection({"collection_name": "test01", "dimension": 128})
        assert status.OK()

        status = gcon.drop_collection("test01")
        assert status.OK()

    @pytest.mark.parametrize("collection", [[], None, 123, {}])
    def test_drop_collection_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.drop_collection(collection)

    def test_drop_collection_non_existent(self, gcon):
        status = gcon.drop_collection("non_existent")
        assert not status.OK()


class TestLoadCollection:
    def test_load_collection_normal(self, gcon, gvector):
        status = gcon.load_collection(gvector)
        assert status.OK()

    @pytest.mark.parametrize("name", [[], bytes(), 123, True, False])
    def test_load_collection_invalid_name(self, name, gcon):
        with pytest.raises(ParamError):
            gcon.load_collection(name)

    def test_load_collection_non_existent(self, gcon):
        status = gcon.load_collection("test_load_collection_non_existent")
        assert not status.OK()

    def test_load_collection_with_partition(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["_default"])
        assert status.OK()

    def test_load_collection_with_partition_non_exist(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["_default", "AQAA"])
        assert status.OK()

    def test_load_collection_with_all_partition_non_exist(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["AQAA"])
        assert not status.OK()

class TestReleaseCollection:
    def test_release_collection_normal(self, gcon, gvector):
        status = gcon.load_collection(gvector)
        assert status.OK()
        status = gcon.release_collection(gvector)
        assert status.OK()

    @pytest.mark.parametrize("name", [[], bytes(), 123, True, False])
    def test_release_collection_invalid_name(self, name, gcon):
        with pytest.raises(ParamError):
            gcon.release_collection(name)

    def test_release_collection_non_existent(self, gcon):
        status = gcon.release_collection("test_release_collection_non_existent")
        assert not status.OK()

    def test_release_collection_with_partition(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["_default"])
        assert status.OK()
        status = gcon.release_collection(gvector, ["_default"])
        assert status.OK()

    def test_release_collection_with_partition_non_exist(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["_default"])
        assert status.OK()
        status = gcon.release_collection(gvector, ["_default", "test_release_collection_with_partition_non_exist"])
        assert status.OK()

    def test_release_collection_with_all_partition_non_exist(self, gcon, gvector):
        status = gcon.load_collection(gvector, ["_default"])
        assert status.OK()
        status = gcon.release_collection(gvector, ["test_release_collection_with_partition_non_exist"])
        assert not status.OK()
