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

    def test_search_async(self, connect, vrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        dsl = vector_dsl(topk, xq, "L2")

        def search_cb(results):
            assert results

        try:
            future0 = connect.search(vrecords, dsl, _async=True)
            future0.result()

            future1 = connect.search(vrecords, dsl, _async=True, _callback=search_cb)
            future1.done()
        except Exception as e:
            pytest.fail(f"{e}")


class TestSearchDSL:
    def test_search_single_vector_dsl(self, connect, vrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        metric = "L2"

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

        try:
            connect.search(vrecords, dsl)
        except Exception as e:
            pytest.fail(f"{e}")

    def test_search_multi_condition_in_single_must(self, connect, ivrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        xi = integer_factory(2)
        metric = "L2"

        dsl = {
            "bool": {
                "must": [
                    {
                        "term": {"Int": xi}
                    },
                    {
                        "vector": {
                            "Vec": {"topk": topk, "query": xq, "metric_type": metric}
                        }
                    }
                ]
            }
        }

        try:
            connect.search(ivrecords, dsl)
        except Exception as e:
            pytest.fail(f"{e}")

    @pytest.mark.skip(reason="Bug in Milvus v0.11.0")
    def test_search_multi_clause(self, connect, ivrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        xi = integer_factory(2)
        metric = "L2"

        dsl = {
            "bool": {
                "must": [
                    {
                        "must": [
                            {
                                "should": [
                                    {
                                        "term": {"Int": xi}
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "must": [
                            {
                                "vector": {
                                    "Vec": {"topk": topk, "query": xq, "metric_type": metric}
                                }
                            }
                        ]
                    }
                ]
            }
        }

        try:
            connect.search(ivrecords, dsl)
        except Exception as e:
            pytest.fail(f"{e}")


class TestSearchInSegment:

    def get_all_segment_id(self, conn, collection, partition=None):
        stats = conn.get_collection_stats(collection)

        segment_ids = list()
        for par in stats['partitions']:
            if partition is not None and partition != par['tag']:
                continue

            segs = par['segments']
            if segs:
                for seg in segs:
                    segment_ids.append(seg['id'])

        return segment_ids

    def test_search_in_segment(self, connect, vrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        dsl = vector_dsl(topk, xq, "L2")

        segment_ids = self.get_all_segment_id(connect, vrecords)

        try:
            connect.search_in_segment(vrecords, segment_ids, dsl)
        except Exception as e:
            pytest.fail(f"{e}")

    def test_search_in_segment_with_some_id_not_exist(self, connect, vrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        dsl = vector_dsl(topk, xq, "L2")

        segment_ids = self.get_all_segment_id(connect, vrecords)
        segment_ids.append(max(segment_ids) + 1)

        try:
            connect.search_in_segment(vrecords, segment_ids, dsl)
        except Exception as e:
            pytest.fail(f"{e}")

    def test_search_in_segment_async(self, connect, vrecords, dim):
        topk = 10
        xq = records_factory(dim, 10)
        dsl = vector_dsl(topk, xq, "L2")

        def search_cb(results):
            assert results

        segment_ids = self.get_all_segment_id(connect, vrecords)

        try:
            future0 = connect.search_in_segment(vrecords, segment_ids, dsl, _async=True)
            future0.result()

            future1 = connect.search_in_segment(vrecords, segment_ids, dsl, _async=True, _callback=search_cb)
            future1.done()
        except Exception as e:
            pytest.fail(f"{e}")
