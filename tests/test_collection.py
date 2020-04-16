import sys
sys.path.append(".")
from unittest import mock

import grpc
import pytest
from grpc._channel import _UnaryUnaryMultiCallable as uum

from milvus import Milvus, MetricType, IndexType, ParamError

from factorys import collection_schema_factory


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
                             [MetricType.L2, MetricType.IP, MetricType.HAMMING, MetricType.JACCARD,
                              MetricType.TANIMOTO, MetricType.SUPERSTRUCTURE, MetricType.SUBSTRUCTURE])
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

    # @mock.patch("grpc._channel._UnaryUnaryMultiCallable")
    # def test_create_collection_exception(self, mock_future, gcon):
    #     mock_future.future = mock.Mock(side_effect=grpc.FutureTimeoutError())
    #     collection_param = {
    #         "collection_name": "test_create_collection_exception",
    #         "dimension": 128,
    #         "metric_type": MetricType.L2,
    #         "index_file_size": 10
    #     }
    #     with pytest.raises(Exception):
    #         gcon.create_collection(collection_param)


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
    def test_describe_collection_normal(self, gcon, gcollection):
        status, info = gcon.describe_collection(gcollection)
        assert status.OK()
        assert info.dimension == 128

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_describe_collection_normal(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.describe_collection(collection)

    def test_describe_collection_non_existent(self, gcon):
        status, _ = gcon.describe_collection("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestCountCollection:
    def test_count_collection_normal(self, gcon, gvector):
        status, count = gcon.count_collection(gvector)
        assert status.OK()
        assert count == 10000

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_count_collection_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.count_collection(collection)

    def test_count_collection_non_existent(self, gcon):
        status, _ = gcon.count_collection("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestCollectinInfo:
    def test_collection_info_normal(self, gcon, gvector):
        status, info = gcon.collection_info(gvector)
        assert status.OK()
        assert info.count == 10000

        par0_stat = info.partitions_stat[0]
        assert par0_stat.tag == "_default"
        assert par0_stat.count == 10000

    @pytest.mark.parametrize("collection", [None, 123, [123], {}])
    def test_count_collection_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.collection_info(collection)

    def test_count_collection_non_existent(self, gcon):
        status, _ = gcon.collection_info("sfsfsfsfsfsfsfsfsfsf")
        assert not status.OK()


class TestShowCollections:
    def test_show_collections(self, gcon, gcollection):
        status, names = gcon.show_collections()
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
        status = gcon.preload_collection(gvector)
        assert status.OK()

    @pytest.mark.parametrize("name", [[], bytes(), 123, True, False])
    def test_load_collection_invalid_name(self, name, gcon):
        with pytest.raises(ParamError):
            gcon.preload_collection(name)

    def test_load_collection_non_existent(self, gcon):
        status = gcon.preload_collection("test_load_collection_non_existent")
        assert not status.OK()
