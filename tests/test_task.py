import random

import pytest

from milvus import ParamError
from factorys import records_factory

dim = 128
nq = 100


class TestFlush:
    def test_flush(self, gcon):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush(collection_list)
        assert status.OK()

        for collection in collection_list:
            gcon.drop_collection(collection)

    def test_flush_with_none(self, gcon, gcollection):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush()
        assert status.OK(), status.message

        for collection in collection_list:
            gcon.drop_collection(collection)

    def test_flush_async_normal(self, gcon, gcollection, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        records = records_factory(dim, nq)
        gcon.insert(gcollection, records)
        future = gcon.flush([gcollection], _async=True)
        status = future.result()
        assert status.OK()

    def test_flush_async_callback(self, gcon, gcollection, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        def cb(status):
            assert status.OK()

        records = records_factory(dim, nq)
        gcon.insert(gcollection, records)
        future = gcon.flush([gcollection], _async=True, _callback=cb)
        future.done()


class TestCompact:
    def test_compact_normal(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.insert(collection_name=gcollection, records=vectors)
        assert status.OK()

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_after_delete(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.insert(collection_name=gcollection, records=vectors)
        assert status.OK(), status.message

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status = gcon.delete_entity_by_id(gcollection, ids[100:1000])
        assert status, status.message

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_async_normal(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        future = gcon.compact(gvector, _async=True)
        status = future.result()
        assert status.OK()

    def test_compact_async_callback(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        def cb(status):
            assert status.OK()

        future = gcon.compact(gvector, _async=True, _callback=cb)
        future.done()

    def test_compact_with_empty_collection(self, gcon, gcollection):
        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_with_non_exist_name(self, gcon):
        status = gcon.compact(collection_name="die333")
        assert not status.OK()

    @pytest.mark.parametrize("name", [123, None, [123], True, {}])
    def test_compact_with_invalid_name(self, name, gcon):
        with pytest.raises(ParamError):
            gcon.compact(collection_name=name)


class TestCmd:
    def test_cmd_version(self, gcon):
        status, reply = gcon.server_version()
        assert status.OK()

    @pytest.mark.parametrize("cmd", ["version", "status", "mode", "tasktable", "get_system_info", "build_commit_id"])
    def test_cmd_whole(self, cmd, gcon):
        status, reply = gcon._cmd(cmd)
        assert status.OK()


class TestConfig:
    def test_get_config(self, gcon):
        status, value = gcon.get_config("cache", "cache_size")
        assert status.OK()

    def test_set_config(self, gcon):
        status, value = gcon.get_config("cache", "cache_size")
        assert status.OK()

        status, reply = gcon.set_config("cache", "cache_size", value)
        assert status.OK()

