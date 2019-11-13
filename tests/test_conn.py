import sys

sys.path.append(".")

import pytest

from milvus import Milvus


class TestConn:

    @pytest.mark.repeat(20)
    @pytest.mark.parametrize("url", [
        # "tcp:// :19530",
        "tcp://123.0.0.1:19530",
        # "tcp://127.0.0:19530",
        # "tcp://255.0.0.0:19530",
        # "tcp://255.255.0.0:19530",
        # "tcp://255.255.255.0:19530",
        # "tcp://255.255.255.255:19530",
        # "tcp://\n:19530"
    ])
    def test_conn(self, url):
        with pytest.raises(Exception):
            client = Milvus()
            client.connect(uri=url)
