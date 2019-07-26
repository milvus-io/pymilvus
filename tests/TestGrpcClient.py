import logging
import time
from milvus.client.GrpcClient import Prepare, GrpcMilvus
from milvus.client.Abstract import IndexType
from factorys import *

LOGGER = logging.getLogger(__name__)

dim = 16

class TestPing:

    def test_ping_server_version(self):
        milvus = GrpcMilvus()
        milvus.connect()

        _, version = milvus.server_version()
        assert version == '0.3.1'


class TestCreateTable:
    
    def test_create_table_normal(self, gcon):
        param = table_schema_factory()

        status = gcon.create_table(param)
        assert status.OK()

    def test_create_table_name_wrong(self, gcon):
        param = table_schema_factory()
        param['table_name'] = '.....'
        status = gcon.create_table(param)
        LOGGER.error(status)
        assert not status.OK()
        

class TestDescribTable:

    def test_describe_table_normal(self, gcon):
        param = table_schema_factory()
        gcon.create_table(param)

        status, table = gcon.describe_table(param['table_name'])
        assert status.OK()
        assert table.table_name == param['table_name']

        status, table = gcon.describe_table('table_not_exists')
        assert not status.OK()


class TestShowTables:
    def test_show_tables_normal(self, gcon):
        status, tables = gcon.show_tables()
        assert status.OK()


class TestDeleteTable:
    def test_delete_table_normal(self, gcon):
        param = table_schema_factory()
        s = gcon.create_table(param)
        _, tables = gcon.show_tables()
        assert param['table_name'] in tables
        

        status = gcon.delete_table(param['table_name'])
        _, tables = gcon.show_tables()
        assert param['table_name'] not in tables


class TestHasTable:
    def test_has_table(self, gcon):
        param = table_schema_factory()
        s = gcon.create_table(param)

        flag = gcon.has_table(param['table_name'])
        assert flag


class TestAddVectors:
    
    def test_add_vectors_normal(self, gcon, gtable):
        vectors = records_factory(dim)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == 20

        time.sleep(2)
        
        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()
        assert count == 20


class TestSearchVectors:
    def test_search_vectors_normal_1_with_ranges(self, gcon, gtable):
        vectors = records_factory(dim)
        status, ids = gcon.add_vectors(gtable, vectors)
        
        ranges = ranges_factory()
        time.sleep(2)

        s_vectors = [vectors[0]]

        status, result = gcon.search_vectors(gtable, 1, s_vectors, ranges)
        assert status.OK()
        assert len(result) == 1
        assert len(result[0]) == 1

