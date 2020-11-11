import pytest

from factorys import fake, records_factory, integer_factory


def vector_dsl(topk, xq):
    dsl = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "Vec": {"topk": topk, "query": xq, "metric_type": "L2"}
                    }
                }
            ]
        }
    }

    return dsl


class TestSearch:
    def test_search_flat(self, connect, vrecords, dim):
        query_vectors = records_factory(dim, 10)

        dsl = vector_dsl(10, query_vectors)

        results = connect.search(vrecords, dsl)
        assert len(results) == 10
        assert len(results[0]) == 10

    def test_search_scalar_flat(self, connect, ivrecords, dim):
        query_vectors = records_factory(dim, 10)

        dsl = vector_dsl(10, query_vectors)

        results = connect.search(ivrecords, dsl)
        assert len(results) == 10
        assert len(results[0]) == 10

    def test_search_with_empty_collection(self, connect, vcollection, dim):
        query_vectors = records_factory(dim, 10)

        dsl = vector_dsl(10, query_vectors)

        results = connect.search(vcollection, dsl)
        assert len(results) == 10
        for result in results:
            assert len(result) == 0

    @pytest.mark.parametrize("par", ["_default", "par_0"])
    def test_search_with_same_vectors(self, par, connect, vcollection, dim):
        xb = records_factory(dim, 50000)
        bulk_entities = [
            {"name": "Vec", "values": xb}
        ]

        if par != "_default":
            connect.create_partition(vcollection, par)

        ids = connect.bulk_insert(vcollection, bulk_entities, partition_tag=par)
        connect.flush([vcollection])

        dsl = vector_dsl(10, xb[: 2])
        results = connect.search(vcollection, dsl)

        assert len(results) == 2
        for id_, result in zip(ids[: 2], results):
            assert len(result) == 10
            assert result[0].id == id_
            assert result[0].distance < 1e-5

    def test_search_with_fields(self, connect, ivrecords, dim):
        xq = records_factory(dim, 10)
        dsl = vector_dsl(10, xq)

        results = connect.search(ivrecords, dsl, fields=["Int", "Vec"])
        assert len(results) == 10
        assert len(results[0]) == 10
        assert isinstance(results[0][0].entity.get("Int"), int)
        assert len(results[0][0].entity.get("Vec")) == dim
