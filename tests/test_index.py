import time

import pytest

from milvus import BaseError


class TestCreateIndex:
    @pytest.mark.parametrize("index, param", [("IVF_FLAT", {"nlist": 512}),
                                              ("IVF_PQ", {"nlist": 512, "m": 8}),
                                              ("IVF_SQ8", {"nlist": 512}),
                                              ("IVF_SQ8_HYBRID", {"nlist": 512}),
                                              ("ANNOY", {"n_trees": 8}),
                                              ("HNSW", {"M": 16, "efConstruction": 40}),
                                              ("NSG", {"search_length": 60,
                                                       "out_degree": 30,
                                                       "candidate_pool_size": 300,
                                                       "knng": 50})])
    def test_create_index_whole(self, index, param, connect, vrecords):
        mode_ = connect._cmd("mode")
        if mode_ == "CPU" and index in ("IVF_SQ8_HYBRID",):
            pytest.skip(f"Index {index} not support in CPU version")

        if mode_ == "GPU" and index in ():
            pytest.skip(f"Index {index} not support in GPU version")

        connect.create_index(vrecords, "Vec", {"index_type": index, "metric_type": "L2", "params": param})

    @pytest.mark.parametrize("index, param", [("IVF_PQ", {"nlist": 512, "m": 8}),
                                              ("HNSW", {"M": 16, "efConstruction": 40}),
                                              ("NSG", {"search_length": 60,
                                                       "out_degree": 30,
                                                       "candidate_pool_size": 300,
                                                       "knng": 50})])
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

    def test_create_index_scalar_index(self, connect, ivrecords):
        try:
            connect.create_index(ivrecords, "Int", {"index_type": "SORTED"})
        except Exception as e:
            pytest.fail(f"{e}")

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


class TestIndexInfo:
    def test_index_info_with_flat_index(self, connect, vrecords):
        info = connect.get_collection_info(vrecords)
        for field in info['fields']:
            if field['name'] == 'Vec':
                indexes = field['indexes'][0]
                assert not indexes
                assert isinstance(indexes, dict)

    def test_index_info_normal(self, connect, vrecords):
        connect.create_index(vrecords, "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 512}})
        info = connect.get_collection_info(vrecords)
        for field in info['fields']:
            if field['name'] == 'Vec':
                indexes = field['indexes'][0]
                assert indexes['index_type'] == 'IVF_FLAT'
                assert indexes['metric_type'] == 'L2'
                assert indexes['params'] == {'nlist': 512}

    def test_index_info_with_scalar_index(self, connect, ivrecords):
        try:
            connect.create_index(ivrecords, "Int", {"index_type": "SORTED"})
            info = connect.get_collection_info(ivrecords)
            for field in info['fields']:
                if field['name'] == 'Int':
                    indexes = field['indexes'][0]
                    assert indexes['index_type'] == 'SORTED'
        except Exception as e:
            pytest.fail(f"{e}")


class TestDropIndex:
    def test_drop_index_flat(self, connect, vrecords):
        connect.drop_index(vrecords, "Vec")

    def test_drop_index_with_empty_collection(self, connect, vcollection):
        connect.drop_index(vcollection, "Vec")

    def test_drop_index_after_crated(self, connect, vrecords):
        connect.create_index(vrecords, "Vec", {"index_type": "IVF_FLAT",
                                               "metric_type": "L2",
                                               "params": {"nlist": 512}})
        time.sleep(0.1)

        connect.drop_index(vrecords, "Vec")

        info = connect.get_collection_info(vrecords)

        for field in info['fields']:
            if field['name'] == 'Vec':
                indexes = field['indexes'][0]
                assert not indexes
                assert isinstance(indexes, dict)

