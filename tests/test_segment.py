import pytest

from milvus import ParamError


class TestSegment:
    def test_get_segment_ids(self, gcon, gvector):
        status, info = gcon.collection_info(gvector)
        assert status.OK()
        # import pdb;pdb.set_trace()
        seg0 = info["partitions"][0]["segments"][0]

        status, ids = gcon.get_vector_ids(gvector, seg0["name"])
        assert status.OK(), status.message
        assert isinstance(ids, list)
        assert len(ids) == 10000

    @pytest.mark.parametrize("collection", [123, None, [], {}, "", True, False])
    @pytest.mark.parametrize("segment", [123, None, [], {}, "", True, False])
    def test_get_segment_invalid_param(self, collection, segment, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_ids(collection, segment)

    def test_get_segment_non_existent_collection_segment(self, gcon, gcollection):
        status, _ = gcon.get_vector_ids("ijojojononsfsfswgsw", "aaa")
        assert not status.OK()

        status, _ = gcon.get_vector_ids(gcollection, "aaaaaa")
        assert not status.OK()
