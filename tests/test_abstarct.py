import sys

sys.path.append(".")

from milvus.client.Abstract import *
import pytest


class TestConnectIntf:
    intf = ConnectIntf()

    def test_connect(self):
        with pytest.raises(NotImplementedError):
            self.intf.connect()

    def test_connected(self):
        with pytest.raises(NotImplementedError):
            self.intf.connected()

    def test_disconnect(self):
        with pytest.raises(NotImplementedError):
            self.intf.disconnect()

    def test_create_table(self):
        with pytest.raises(NotImplementedError):
            self.intf.create_table(None)

    def test_has_table(self):
        with pytest.raises(NotImplementedError):
            self.intf.has_table("")

    def test_delete_table(self):
        with pytest.raises(NotImplementedError):
            self.intf.delete_table("")

    def test_add_vectors(self):
        with pytest.raises(NotImplementedError):
            self.intf.add_vectors("", [])

    def test_search_vectors(self):
        with pytest.raises(NotImplementedError):
            self.intf.search_vectors(None, None, None, None)

    def test_search_vectors_in_files(self):
        with pytest.raises(NotImplementedError):
            self.intf.search_vectors_in_files(None, None, None, None, None)

    def test_describe_table(self):
        with pytest.raises(NotImplementedError):
            self.intf.describe_table(None)

    def test_get_table_row_count(self):
        with pytest.raises(NotImplementedError):
            self.intf.get_table_row_count(None)

    def test_show_tables(self):
        with pytest.raises(NotImplementedError):
            self.intf.show_tables()

    def test_create_index(self):
        with pytest.raises(NotImplementedError):
            self.intf.create_index('', None)

    def test_client_version(self):
        with pytest.raises(NotImplementedError):
            self.intf.client_version()

    def test_server_status(self):
        with pytest.raises(NotImplementedError):
            self.intf.server_status()

    def test_server_version(self):
        with pytest.raises(NotImplementedError):
            self.intf.server_version()

    def test_delete_vectors_by_range(self):
        with pytest.raises(NotImplementedError):
            self.intf.delete_vectors_by_range(None, None)

    def test_preload_table(self):
        with pytest.raises(NotImplementedError):
            self.intf.preload_table("")

    def test_describe_index(self):
        with pytest.raises(NotImplementedError):
            self.intf.describe_index("")

    def test_drop_index(self):
        with pytest.raises(NotImplementedError):
            self.intf.drop_index("")


class TestTableSchema:
    def test_init(self):
        with pytest.raises(ParamError):
            TableSchema(None, None, None, None)

        with pytest.raises(ParamError):
            TableSchema("abc", -1, None, None)

        with pytest.raises(ParamError):
            TableSchema("abc", 1, 1, None)

        table_schema = TableSchema("pymilvus", 1, 1, 1)
        print(table_schema)
