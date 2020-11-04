import grpc
import pytest

from unittest import mock

from grpc._channel import _UnaryUnaryMultiCallable as Uum

from milvus import Milvus, DataType, BaseError

from factorys import fake, records_factory, integer_factory
from utils import MockGrpcError


class TestBulkInsert:

    def test_bulk_insert_normal(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors}
        ]

        try:
            connect.bulk_insert(vcollection, entities)
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_normal2(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(10000)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        try:
            connect.bulk_insert(hvcollection, entities)
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_with_id(self, connect, vicollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors}
        ]
        ids = [i for i in range(10000)]

        try:
            connect.bulk_insert(vicollection, entities, ids)
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_with_type(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR}
        ]

        try:
            connect.bulk_insert(vcollection, entities)
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(e))

    def test_bulk_insert_with_wrong_type(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.BINARY_VECTOR}
        ]

        with pytest.raises(TypeError):
            connect.bulk_insert(vcollection, entities)

    # @pytest.mark.parametrize("sd", [DataType.INT32, DataType.FLOAT, DataType.DOUBLE])
    @pytest.mark.parametrize("sd", [DataType.INT32, DataType.FLOAT])
    def test_bulk_insert_with_wrong_scalar_type(self, sd, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(10000)

        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR},
            {"name": "Int", "values": integers, "type": sd}
        ]

        with pytest.raises(BaseError):
            connect.bulk_insert(hvcollection, entities)

    def test_bulk_insert_with_wrong_scalar_type_then_correct_type(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(10000)

        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR},
            {"name": "Int", "values": integers, "type": DataType.FLOAT}
        ]

        with pytest.raises(BaseError):
            connect.bulk_insert(hvcollection, entities)

        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR},
            {"name": "Int", "values": integers, "type": DataType.INT64}
        ]

        try:
            connect.bulk_insert(hvcollection, entities)
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_with_collection_non_exist(self, connect, hvcollection, dim):
        collection = hvcollection + "_c1"
        vectors = records_factory(dim, 1)
        integers = integer_factory(1)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        with pytest.raises(BaseError):
            connect.bulk_insert(collection, entities)

    def test_create_collection_exception(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 1)
        integers = integer_factory(1)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        mock_grpc_timeout = mock.MagicMock(side_effect=grpc.FutureTimeoutError())
        with mock.patch.object(Uum, 'future', mock_grpc_timeout):
            with pytest.raises(grpc.FutureTimeoutError):
                connect.bulk_insert(hvcollection, entities)

        mock_grpc_error = mock.MagicMock(side_effect=MockGrpcError())
        with mock.patch.object(Uum, 'future', mock_grpc_error):
            with pytest.raises(grpc.RpcError):
                connect.bulk_insert(hvcollection, entities)

        mock_exception = mock.MagicMock(side_effect=Exception("error"))
        with mock.patch.object(Uum, 'future', mock_exception):
            with pytest.raises(Exception):
                connect.bulk_insert(hvcollection, entities)


class TestBulkInsertAsync:
    def test_bulk_insert_async_normal(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors}
        ]

        try:
            ft = connect.bulk_insert(vcollection, entities, _async=True)
            ft.result()
            ft.done()
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_async_normal2(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(10000)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        try:
            ft = connect.bulk_insert(hvcollection, entities, _async=True)
            ft.result()
            ft.done()
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_async_callback(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [
            {"name": "Vec", "values": vectors}
        ]

        def icb(ids):
            if len(ids) != 10000:
                raise ValueError("Result id len is not equal to 10000")

        try:
            ft = connect.bulk_insert(vcollection, entities, _async=True, _callback=icb)
            ft.result()
            ft.done()
        except Exception as e:
            pytest.fail("Unexpected MyError: {}".format(str(e)))

    def test_bulk_insert_async_with_collection_non_exist(self, connect, hvcollection, dim):
        collection = hvcollection + "_c1"
        vectors = records_factory(dim, 1)
        integers = integer_factory(1)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        with pytest.raises(BaseError):
            future = connect.bulk_insert(collection, entities, _async=True)
            future.result()

        entities = [
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR},
            {"name": "Int", "values": integers, "type": DataType.INT64}
        ]

        future = connect.bulk_insert(collection, entities, _async=True)
        with pytest.raises(BaseError):
            future.result()
