import random

import grpc
import pytest

from unittest import mock

from grpc._channel import _UnaryUnaryMultiCallable as Uum

from milvus import Milvus, DataType, BaseError, ParamError
from milvus.client.exceptions import CollectionNotExistException

from factorys import fake, records_factory, integer_factory, binary_records_factory
from utils import MockGrpcError


class TestInsert:
    def test_insert_normal(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [{"Vec": vector} for vector in vectors]

        try:
            connect.insert(vcollection, entities)
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    @pytest.mark.parametrize("scalar", [DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE])
    @pytest.mark.parametrize("vec", [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR])
    def test_insert_scalar(self, scalar, vec, connect, dim):
        collection_name = fake.collection_name()
        collection_param = {
            "fields": [
                {"name": "scalar", "type": scalar},
                {"name": "Vec", "type": vec, "params": {"dim": 128}}
            ]
        }

        try:
            connect.create_collection(collection_name, collection_param)
        except Exception as e:
            pytest.fail(f"Create collection {collection_name} fail: {e}")

        if scalar in (DataType.INT32, DataType.INT64):
            scalars = [random.randint(0, 10000) for _ in range(10000)]
        else:
            scalars = [random.random() for _ in range(10000)]

        if vec in (DataType.FLOAT_VECTOR,):
            vectors = records_factory(dim, 10000)
        else:
            vectors = binary_records_factory(dim, 10000)

        entities = [{"Vec": vector, "scalar": s} for vector, s in zip(vectors, scalars)]

        try:
            connect.insert(collection_name, entities)
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")
        finally:
            connect.drop_collection(collection_name)

    @pytest.mark.parametrize("scalar", [DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE])
    @pytest.mark.parametrize("vec", [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR])
    def test_insert_scalar_with_id(self, scalar, vec, connect, dim):
        collection_name = fake.collection_name()
        collection_param = {
            "fields": [
                {"name": "scalar", "type": scalar},
                {"name": "Vec", "type": vec, "params": {"dim": 128}}
            ],
            "auto_id": False
        }

        try:
            connect.create_collection(collection_name, collection_param)
        except Exception as e:
            pytest.fail(f"Create collection {collection_name} fail: {e}")

        if scalar in (DataType.INT32, DataType.INT64):
            scalars = [random.randint(0, 10000) for _ in range(10000)]
        else:
            scalars = [random.random() for _ in range(10000)]

        if vec in (DataType.FLOAT_VECTOR,):
            vectors = records_factory(dim, 10000)
        else:
            vectors = binary_records_factory(dim, 10000)

        entities = [{"Vec": vector, "scalar": s} for vector, s in zip(vectors, scalars)]
        for i, e in enumerate(entities):
            e["_id"] = i

        try:
            connect.insert(collection_name, entities)
        except Exception as e:
            pytest.fail(f"Unexpected Insert Error: {e}")
        finally:
            connect.drop_collection(collection_name)

    def test_insert_with_collection_non_exist(self, connect):
        collection_name = fake.collection_name()

        with pytest.raises(CollectionNotExistException):
            connect.insert(collection_name, [{"A": 1, "V": [1]}])

    def test_insert_with_wrong_type_data(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(10000)

        entities = [{"Vec": v, "Int": i} for v, i in zip(vectors, integers)]

        connect.insert(hvcollection, entities)

    def test_insert_with_wrong_nonaligned_data(self, connect, hvcollection, dim):
        vectors = records_factory(dim, 10000)
        integers = integer_factory(9998)

        entities = [{"Vec": v, "Int": i} for v, i in zip(vectors[:9998], integers[:9998])]
        entities.append({"Vec": vectors[9998]})

        with pytest.raises(BaseError):
            connect.insert(hvcollection, entities)

    def test_insert_with_mismatch_field(self, connect, hvcollection, dim):

        entities = [{"Vec": records_factory(dim, 1)[0], "Int": 1, "Float": 0.2}]

        with pytest.raises(ParamError):
            connect.insert(hvcollection, entities)

        entities = [{"Vec": records_factory(dim, 1)[0]}]
        with pytest.raises(ParamError):
            connect.insert(hvcollection, entities)

    def test_insert_with_exception(self, connect, hvcollection, dim):
        insert_entities = [{"Int": 10, "Vec": records_factory(dim, 1)[0]}]

        mock_grpc_timeout = mock.MagicMock(side_effect=grpc.FutureTimeoutError())
        with mock.patch.object(Uum, 'future', mock_grpc_timeout):
            with pytest.raises(grpc.FutureTimeoutError):
                connect.insert(hvcollection, insert_entities)

        mock_grpc_error = mock.MagicMock(side_effect=MockGrpcError())
        with mock.patch.object(Uum, 'future', mock_grpc_error):
            with pytest.raises(grpc.RpcError):
                connect.insert(hvcollection, insert_entities)

        mock_exception = mock.MagicMock(side_effect=Exception("error"))
        with mock.patch.object(Uum, 'future', mock_exception):
            with pytest.raises(Exception):
                connect.insert(hvcollection, insert_entities)


class TestInsertAsync:
    def test_insert_async_normal(self, connect, vcollection, dim):
        vectors = records_factory(dim, 10000)
        entities = [{"Vec": vector} for vector in vectors]

        try:
            future = connect.insert(vcollection, entities, _async=True)
            ids = future.result()
            future.done()
            assert len(ids) == 10000
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    def test_insert_async_callback(self, connect, vcollection, dim):
        def cb(inserted_ids):
            if len(inserted_ids) != 10000:
                raise ValueError("The len of result ids is not equal to 10000")

        vectors = records_factory(dim, 10000)
        entities = [{"Vec": vector} for vector in vectors]

        try:
            future = connect.insert(vcollection, entities, _async=True, _callback=cb)
            future.done()
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    def test_insert_async_with_collection_non_exist(self, connect):
        collection_name = fake.collection_name()

        with pytest.raises(BaseError):
            connect.insert(collection_name, {"A": 1, "V": [1]}, _async=True)
