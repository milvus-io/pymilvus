import pytest
from milvus import Milvus, DataType, BaseException

from factorys import fake, records_factory, integer_factory


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
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers, "type": sd}
        ]

        with pytest.raises(BaseException):
            connect.bulk_insert(hvcollection, entities)

    def test_bulk_insert_with_collecton_non_exist(self, connect, hvcollection, dim):
        collection = hvcollection + "_c1"
        vectors = records_factory(dim, 1)
        integers = integer_factory(1)

        entities = [
            {"name": "Vec", "values": vectors},
            {"name": "Int", "values": integers}
        ]

        with pytest.raises(BaseException):
            connect.bulk_insert(collection, entities)
