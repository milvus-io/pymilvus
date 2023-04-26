from pymilvus.client import utils


class TestUtils:
    def test_get_server_type(self):
        urls_and_wants = [
            ('in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com', 'zilliz'),
            ('something.notzillizcloud.com', 'milvus'),
            ('something.zillizcloud.not.com', 'milvus')
        ]
        for (url, want) in urls_and_wants:
            assert utils.get_server_type(url) == want
