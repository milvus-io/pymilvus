import copy
import time
import random
import mock

import sys

sys.path.append('.')
from milvus import ParamError, Milvus

import pytest

client = Milvus()
client._handler.connected = mock.Mock(return_value=True)
client.connected = mock.Mock(return_value=True)


def test_create_table_param(gcon):
    _PARAM = {
        "table_name": "table_name_{}".format(str(random.randint(0, 10000))),
        "dimension": 128
    }

    table_param = copy.deepcopy(_PARAM)

    status = gcon.create_table(table_param)
    assert status.OK()
    time.sleep(1)
    gcon.drop_table(table_param["table_name"])

    table_param["table_name"] = 12343

    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = copy.deepcopy(_PARAM)
    table_param["dimension"] = 'eesfst'
    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = copy.deepcopy(_PARAM)
    table_param["index_file_size"] = -1
    status = gcon.create_table(table_param)
    assert not status.OK()

    table_param = copy.deepcopy(_PARAM)
    table_param["metric_type"] = 0
    with pytest.raises(ParamError):
        gcon.create_table(table_param)


def test_drop_table_param():
    table_name = 124
    client.connected = mock.Mock(return_value=True)
    with pytest.raises(ParamError):
        client.has_table(table_name)


def test_create_index_param():
    index_pram = {
        "nlist": 4096
    }

    with pytest.raises(ParamError):
        client.create_index("test", index_type=0, params=index_pram)

    with pytest.raises(ParamError):
        client.create_index("test", index_type=-1, params=index_pram)

    with pytest.raises(ParamError):
        client.create_index("test", index_type=100, params=index_pram)


class TestInsertParam:
    vectors = [[random.random() for _ in range(16)] for _ in range(10)]

    @pytest.mark.parametrize("table", [None, "", 123, False])
    def test_insert_with_wrong_table_name(self, table):
        with pytest.raises(ParamError):
            client.insert(table, self.vectors)

    @pytest.mark.parametrize("ids", ["3423", 134, [1, 2, 3], False])
    def test_insert_with_wrong_ids(self, ids):
        with pytest.raises(ParamError):
            client.insert("test", self.vectors, ids)

    @pytest.mark.parametrize("part", [1, (), False, []])
    def test_insert_with_wrong_partition(self, part):
        with pytest.raises(ParamError):
            client.insert("test", self.vectors, partition_tag=part)


class TestSearchParam:
    query_vectors = [[random.random() for _ in range(16)] for _ in range(10)]

    @pytest.mark.parametrize("tags", ["", 1, False, [123]])
    def test_search_with_wrong_parittion_args(self, tags):
        with pytest.raises(ParamError):
            client.search("test", top_k=1, query_records=self.query_vectors, partition_tags=tags)
