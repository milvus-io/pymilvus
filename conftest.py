import time

# Third party imports
import pytest

# local application imports
from factorys import gen_unique_str, fake, records_factory
from milvus import Milvus, IndexType, MetricType

default_host = "127.0.0.1"
default_grpc_port = 19530
default_http_port = 19121


def pytest_addoption(parser):
    parser.addoption("--ip", action="store", default=default_host)
    parser.addoption("--handler", action="store", default="GRPC")
    parser.addoption("--port", action="store", default=default_grpc_port)


@pytest.fixture(scope="module")
def gip(request):
    ip_ = request.config.getoption("--ip")
    port_ = request.config.getoption("--port")

    return ip_, port_


@pytest.fixture(scope="module")
def connect(request):
    ip = request.config.getoption("--ip")
    handler = request.config.getoption("--handler")
    port_default = default_http_port if handler == "HTTP" else default_grpc_port
    port = request.config.getoption("--port", default=port_default)

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
    milvus = Milvus()
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
             'index_type': IndexType.IVFLAT,
             'metric_type': MetricType.L2
             }
    connect.create_table(param)

    def teardown():
        status, table_names = connect.show_tables()
        connect.delete_table(table_name)

    request.addfinalizer(teardown)

    return table_name


@pytest.fixture(scope="function")
def gtable(request, gcon):
    table_name = fake.table_name()
    dim = getattr(request.module, "dim", 128)

    param = {'table_name': table_name,
             'dimension': dim,
             'index_file_size': 1024,
             'metric_type': MetricType.L2
             }
    gcon.create_table(param)
    gcon.flush([table_name])

    def teardown():
        status, table_names = gcon.show_tables()
        for name in table_names:
            gcon.drop_table(name)

    request.addfinalizer(teardown)

    return table_name


@pytest.fixture(scope='function')
def gvector(request, gcon, gtable):
    dim = getattr(request.module, 'dim', 128)

    records = records_factory(dim, 10000)

    gcon.insert(gtable, records)
    gcon.flush([gtable])

    return gtable
