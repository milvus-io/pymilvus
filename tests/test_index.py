import pytest

from milvus import BaseError


class TestCreateIndex:
    @pytest.mark.parametrize("index, param",
                             [
                                 ("IVF_FLAT", {"nlist": 512}),
                                 ("IVF_PQ", {"nlist": 512, "m": 8}),
                                 ("IVF_SQ8", {"nlist": 512}),
                                 ("IVF_SQ8_HYBRID", {"nlist": 512}),
                                 ("ANNOY", {"n_trees": 8}),
                                 ("HNSW", {"M": 16, "efConstruction": 40}),
                                 ("NSG", {"search_length": 60,
                                           "out_degree": 30,
                                           "candidate_pool_size": 300,
                                           "knng": 50})
                             ])
    def test_create_index_whole(self, index, param, connect, vrecords):
        mode_ = connect._cmd("mode")
        if mode_ == "CPU" and index in ("IVF_SQ8_HYBRID",):
            pytest.skip(f"Index {index} not support in CPU version")

        if mode_ == "GPU" and index in ():
            pytest.skip(f"Index {index} not support in GPU version")

        connect.create_index(vrecords, "Vec", {"index_type": index, "metric_type": "L2", "params": param})

    @pytest.mark.parametrize("index, param",
                             [
                                 ("IVF_PQ", {"nlist": 512, "m": 8}),
                                 ("HNSW", {"M": 16, "efConstruction": 40}),
                                 ("NSG", {"search_length": 60,
                                          "out_degree": 30,
                                          "candidate_pool_size": 300,
                                          "knng": 50})
                             ])
    def test_create_index_with_IP(self, index, param, connect, vrecords):
        mode_ = connect._cmd("mode")
        if mode_ == "CPU" and index in ("IVF_SQ8_HYBRID",):
            pytest.skip(f"Index {index} not support in CPU version")

        if mode_ == "GPU" and index in ():
            pytest.skip(f"Index {index} not support in GPU version")

        connect.create_index(vrecords, "Vec", {"index_type": index, "metric_type": "IP", "params": param})

    @pytest.mark.parametrize("index", ["BIN_IVF_FLAT"])
    @pytest.mark.parametrize("metric", ["JACCARD", "HAMMING", "TANIMOTO"])
    def test_create_bin_index_whole(self, index, metric, connect, bvrecords):
        try:
            connect.create_index(bvrecords, "Vec",
                                 {"index_type": index, "metric_type": metric, "params": {"nlist": 512}})
        except Exception as e:
            pytest.fail(f"Create index {index} with metric {metric} failed: {e}")

    @pytest.mark.parametrize("index", ["BIN_IVF_FLAT"])
    @pytest.mark.parametrize("metric", ["SUPERSTRUCTURE", "SUBSTRUCTURE"])
    def test_create_bin_index_mismatch_metric(self, index, metric, connect, bvrecords):
        with pytest.raises(BaseError):
            connect.create_index(bvrecords, "Vec",
                                 {"index_type": index, "metric_type": metric, "params": {"nlist": 512}})

    def test_create_index_with_invalid_index_name(self, connect, vrecords):
        with pytest.raises(BaseError):
            connect.create_index(vrecords, "Vec",
                                 {"index_type": "IVF_XXX", "metric_type": "L2", "params": {"nlist": 512}})

    def test_create_index_with_invalid_field(self, connect, vrecords):
        with pytest.raises(BaseError):
            connect.create_index(vrecords, "VecXXX",
                                 {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 512}})

    def test_create_index_getting_info(self, connect, vrecords):
        try:
            connect.create_index(vrecords, "Vec",
                                 {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 512}})
            info = connect.get_collection_info(vrecords)
            for f in info["fields"]:
                if f['name'] == "Vec":
                    indexes = f['indexes']
                    print(indexes)
                    assert indexes[0]['index_type'] == 'IVF_FLAT'
                    assert indexes[0]['params'] == {'nlist': 512}
        except Exception as e:
            pytest.fail(f"Create index or get collection info failed: {e}")
