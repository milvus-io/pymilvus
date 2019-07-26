# Third party imports
import pytest

# local application imports
from factorys import *
from milvus import Milvus, IndexType
from milvus.client.GrpcClient import GrpcMilvus


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
def gcon(request):
    ip = request.config.getoption("--ip")
    port = request.config.getoption("--port")
    milvus = GrpcMilvus()
    milvus.connect(host=ip, port=port)

    def fin():
        try:
            milvus.disconnect()
        except Exception as e:
            print(e)
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
        connect.delete_table(table_name)

    request.addfinalizer(teardown)

    return table_name


@pytest.fixture(scope="function")
def gtable(request, gcon):
    table_name = fake.table_name()
    dim = getattr(request.module, "dim")

    param = {'table_name': table_name,
             'dimension': dim,
             'index_type': IndexType.FLAT,
             'store_raw_vector': False}
    gcon.create_table(param)

    def teardown():
        status, table_names = gcon.show_tables()
        gcon.delete_table(table_name)

    request.addfinalizer(teardown)

    return table_name
