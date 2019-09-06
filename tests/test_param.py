import sys

sys.path.append('.')

from milvus.client.GrpcClient import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType, TableSchema, TopKQueryResult, MetricType
from milvus.client.Exceptions import *

import pytest
import time


def test_create_table_param(gcon):
    _PARAM = {
        "table_name": "table_name_123",
        "dimension": 128
    }

    table_param = _PARAM

    status = gcon.create_table(table_param)
    assert status.OK()
    time.sleep(1)
    gcon.delete_table(table_param["table_name"])

    table_param["table_name"] = 12343

    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = _PARAM
    table_param["dimension"] = 16385
    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = _PARAM
    table_param["index_file_size"] = -1
    with pytest.raises(ParamError):
        gcon.create_table(table_param)

    table_param = _PARAM
    table_param["metric_type"] = 0
    with pytest.raises(ParamError):
        gcon.create_table(table_param)


def test_create_index_param():
    pass
