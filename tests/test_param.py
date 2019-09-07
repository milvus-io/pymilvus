import sys
import copy

sys.path.append('.')

from milvus.client.GrpcClient import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType, TableSchema, TopKQueryResult, MetricType
from milvus.client.Exceptions import *

import pytest
import time
import random


def test_create_table_param(gcon):
    _PARAM = {
        "table_name": "table_name_{}".format(str(random.randint(0, 10000))),
        "dimension": 128
    }

    table_param = copy.deepcopy(_PARAM)

    status = gcon.create_table(table_param)
    assert status.OK()
    time.sleep(1)
    gcon.delete_table(table_param["table_name"])

    table_param["table_name"] = 12343

    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    # TODO: validate max value in server
    # table_param = copy.deepcopy(_PARAM)
    # table_param["dimension"] = 16385
    # with pytest.raises(ParamError):
    #     gcon.create_table(table_param)

    table_param = copy.deepcopy(_PARAM)
    table_param["dimension"] = 'eesfst'
    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = copy.deepcopy(_PARAM)
    table_param["index_file_size"] = -1
    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = copy.deepcopy(_PARAM)
    table_param["metric_type"] = 0
    with pytest.raises(ParamError):
        gcon.create_table(table_param)


def test_has_table_param(gcon):
    table_name = "test_has_table_param"
    flag = gcon.has_table(table_name)
    assert not flag

    table_name = 124
    with pytest.raises(ParamError):
        gcon.has_table(table_name)


def test_delete_table_param(gcon):
    table_name = "test_delete_table_param"
    flag = gcon.has_table(table_name)
    assert not flag

    table_name = 124
    with pytest.raises(ParamError):
        gcon.has_table(table_name)


def test_create_index_param(gcon, gvector):
    status = gcon.create_index(gvector, None)
    assert status.OK()

    index = {
        'index_type': 0,
        'nlist': 16384
    }

    with pytest.raises(ParamError):
        gcon.create_index(index)

    index = {
        'index_type': 1,
        'nlist': 0
    }
    with pytest.raises(ParamError):
        gcon.create_index(index)


def test_add_vectors_param(gcon, gtable):
    table_name = ""

    vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
    with pytest.raises(ParamError):
        gcon.add_vectors(table_name, vectors)

    ids = [1, 2, 3]
    with pytest.raises(ParamError):
        gcon.add_vectors(table_name, vectors, ids)


def test_search_param(gcon, gvector):
    query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
    with pytest.raises(ParamError):
        gcon.search_vectors(gvector, top_k=0, nprobe=16, query_records=query_vectors)

    # TODO: top_k is not a limit of max value
    # with pytest.raises(ParamError):
    #     gcon.search_vectors(gvector, top_k=5000, nprobe=16, query_records=query_vectors)

    with pytest.raises(ParamError):
        gcon.search_vectors(gvector, top_k=1, nprobe=0, query_records=query_vectors)

    # TODO: nprobe is not  a limit of max value
    # with pytest.raises(ParamError):
    #     gcon.search_vectors(gvector, top_k=1, nprobe=20000, query_records=query_vectors)
