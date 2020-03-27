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