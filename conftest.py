# Third party imports
import pytest

# local application imports
from factorys import *
from milvus import Milvus, IndexType, MetricType

host = "127.0.0.1"


def pytest_addoption(parser):
    parser.addoption("--ip", action="store", default=host)
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
    dim = getattr(request.module, "dim")

    param = {'table_name': table_name,
             'dimension': dim,
             'index_file_size': 1024,
             'metric_type': MetricType.L2
             }
    gcon.create_table(param)

    def teardown():
        status, table_names = gcon.show_tables()
        for name in table_names:
            gcon.delete_table(name)

    request.addfinalizer(teardown)

    return table_name


@pytest.fixture(scope='function')
def gvector(request, gcon, gtable):
    dim = getattr(request.module, 'dim')

    records = records_factory(dim, 10000)

    gcon.add_vectors(gtable, records)
    time.sleep(3)

    return gtable
