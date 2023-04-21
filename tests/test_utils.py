from pymilvus.client import utils


class TestUtils:
    def test_get_server_type(self):
        url1 = 'in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com'
        assert utils.get_server_type(url1) == "milvus"

        url2 = 'https://in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com'
        assert utils.get_server_type(url2) == "zilliz"

        url3 = 'http://in01-0390f61a8675594.aws-us-west-2.vectordb.zillizcloud.com'
        assert utils.get_server_type(url3) == "milvus"

        url4 = 'https://something.notzillizcloud.com'
        assert utils.get_server_type(url4) == "milvus"

        url5 = 'https://something.zillizcloud.not.com'
        assert utils.get_server_type(url5) == "milvus"


