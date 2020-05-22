import pytest

from milvus import ParamError


class TestSegment:
    def test_get_segment_ids(self, gcon, gvector):
        status, info = gcon.get_collection_stats(gvector)
        assert status.OK()
        seg0 = info["partitions"][0]["segments"][0]

        status, ids = gcon.list_id_in_segment(gvector, seg0["name"])
        assert status.OK(), status.message
        assert isinstance(ids, list)
        assert len(ids) == 10000

    @pytest.mark.parametrize("collection", [123, None, [], {}, "", True, False])
    @pytest.mark.parametrize("segment", [123, None, [], {}, "", True, False])
    def test_get_segment_invalid_param(self, collection, segment, gcon):
        with pytest.raises(ParamError):
            gcon.list_id_in_segment(collection, segment)

    def test_get_segment_non_existent_collection_segment(self, gcon, gcollection):
        status, _ = gcon.list_id_in_segment("ijojojononsfsfswgsw", "aaa")
        assert not status.OK()

        status, _ = gcon.list_id_in_segment(gcollection, "aaaaaa")
        assert not status.OK()
