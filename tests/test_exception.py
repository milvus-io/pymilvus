from unittest import mock
import pytest
import grpc

import sys

sys.path.append(".")
from milvus import Milvus
from milvus.client.GrpcClient import GrpcMilvus
from milvus.grpc_gen.milvus_pb2_grpc import MilvusServiceStub as Stub

import random
from unittest import mock
from grpc import Future
from milvus.client.types import Status
from grpc._channel import _Rendezvous


class TestGrpcError:

    client = Milvus()
    client.connect()

    @mock.patch.object(Status, "__init__")
    def test_add_vectors(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.delete_table("Hao3")

    @mock.patch.object(Status, "__init__")
    def test_create_table(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        param = {
            "table_name": "yyhh",
            "dimension": 128
        }

        with pytest.raises(AttributeError):
            self.client.create_table(param)

    @mock.patch.object(Status, "__init__")
    def test_has_table(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        ok = self.client.has_table("param")
        assert not ok

    @mock.patch.object(Status, "__init__")
    def test_delete_table(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.delete_table("param")

    @mock.patch.object(Status, "__init__")
    def test_create_index(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.create_index("param")

    @mock.patch.object(Status, "__init__")
    def test_add_vectors(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        vectors = [[random.random() for _ in range(128)] for _ in range(10)]

        with pytest.raises(AttributeError):
            self.client.add_vectors("tabe", vectors)
    #
    # @mock.patch.object(Status, "__init__")
    # def test_search_vectors(self, mock_result):
    #     mock_result.side_effect = grpc.RpcError()
    #
    #     vectors = [[random.random() for _ in range(128)] for _ in range(10)]
    #
    #     with pytest.raises(AttributeError):
    #         self.client.search_vectors("tabe", 1, 1, vectors)

    # @mock.patch.object(Status, "__init__")
    # def test_search_vectors_in_files(self, mock_result):
    #     mock_result.side_effect = grpc.RpcError()
    #
    #     vectors = [[random.random() for _ in range(128)] for _ in range(10)]
    #
    #     with pytest.raises(AttributeError):
    #         self.client.search_vectors_in_files("tabe", [1], vectors, 1, 1)

    @mock.patch.object(Status, "__init__")
    def test_describe_table(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.describe_table("tabe")

    @mock.patch.object(Status, "__init__")
    def test_show_tables(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.show_tables()

    @mock.patch.object(Status, "__init__")
    def test__cmd(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client._cmd("999")

    @mock.patch.object(Status, "__init__")
    def test_delete_vectors_by_range(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.delete_vectors_by_range("999", "2010-01-01", "2020-12-31")

    @mock.patch.object(Status, "__init__")
    def test_preload_table(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.preload_table("999")

    @mock.patch.object(Status, "__init__")
    def test_describe_index(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.describe_index("999")

    @mock.patch.object(Status, "__init__")
    def test_drop_index(self, mock_result):
        mock_result.side_effect = grpc.RpcError()

        with pytest.raises(AttributeError):
            self.client.drop_index("999")

