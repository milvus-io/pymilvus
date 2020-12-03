import pytest

from factorys import fake, records_factory, integer_factory


def vector_dsl(topk, xq, metric="L2"):
    dsl = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "Vec": {"topk": topk, "query": xq, "metric_type": metric}
                    }
                }
            ]
        }
    }

    return dsl


class TestSearch:
    @pytest.mark.parametrize("metric", ["L2", "IP"])
    def test_search_flat(self, metric, connect, vrecords, dim):
        query_vectors = records_factory(dim, 10)

        dsl = vector_dsl(10, query_vectors, metric)

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

    def test_search_with_partitions(self, connect, vrecords, dim):
        connect.create_partition(vrecords, "p_1")
        connect.create_partition(vrecords, "p_2")

        vectors = records_factory(dim, 10000)

        entities = [
            {"name": "Vec", "values": vectors}
        ]
        ids1 = connect.bulk_insert(vrecords, entities, partition_tag="p_1")
        ids2 = connect.bulk_insert(vrecords, entities, partition_tag="p_2")

        connect.flush([vrecords])

        query_vectors = vectors[: 1]

        topk = 5
        dsl = vector_dsl(topk, query_vectors)

        results = connect.search(vrecords, dsl)
        assert len(results) == 1
        assert len(results[0]) == topk
        assert results[0][0].id == ids1[0]
        assert results[0][0].distance < 1e-5
        assert results[0][1].distance < 1e-5

        results = connect.search(vrecords, dsl, partition_tags=["p_1"])
        assert len(results) == 1
        assert len(results[0]) == topk
        assert results[0][0].id == ids1[0]
        assert results[0][1].distance > results[0][0].distance

        # regex match
        results = connect.search(vrecords, dsl, partition_tags=["p_(.*)"])
        assert len(results) == 1
        assert len(results[0]) == topk
        assert results[0][0].distance < 1e-5
        assert results[0][1].distance < 1e-5

