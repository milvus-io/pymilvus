# Third party imports
import pytest

# local application imports
from factorys import gen_unique_str
from milvus import Milvus, IndexType


def pytest_addoption(parser):
    parser.addoption("--ip", action="store", default="localhost")
    parser.addoption("--port", action="store", default=19530)


@pytest.fixture(scope="module")
def connect(request):
    ip = request.config.getoption("--ip")
    port = request.config.getoption("--port")
    milvus = Milvus()
    milvus.connect(host=ip, port=port)

    def fin():
        try:
            milvus.disconnect()
        except:
            pass

    request.addfinalizer(fin)
    return milvus


@pytest.fixture(scope="module")
def args(request):
    ip = request.config.getoption("--ip")
    port = request.config.getoption("--port")
    args = {"ip": ip, "port": port}
    return args


@pytest.fixture(scope="function")
def table(request, connect):
    ori_table_name = getattr(request.module, "table_id", "test")
    table_name = gen_unique_str(ori_table_name)
    dim = getattr(request.module, "dim", "128")
    param = {'table_name': table_name,
             'dimension': dim,
             'index_type': IndexType.FLAT,
             'store_raw_vector': False}
    connect.create_table(param)

    def teardown():
        status, table_names = connect.show_tables()
        for table_name in table_names:
            connect.delete_table(table_name)

    request.addfinalizer(teardown)

    return table_name
