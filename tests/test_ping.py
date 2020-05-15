import pytest

from milvus import Milvus


@pytest.mark.skip
class TestPing:
    def test_ping_normal(self, gip):
        client = Milvus(*gip)
        assert client.ping()

    @pytest.mark.parametrize("ip", ["aa", 123, True])
    @pytest.mark.parametrize("port", ["bbb", False])
    def test_ping_invalid_addr(self, ip, port):
        with pytest.raises(Exception):
            client = Milvus(ip, port)
            client.ping()
