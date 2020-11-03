
# Third party imports
import pytest

# local application imports
from factorys import gen_unique_str, fake, records_factory, integer_factory
from milvus import Milvus, DataType


default_host = "127.0.0.1"
default_grpc_port = 19530
default_http_port = 19121
default_vector_dim = 128


Field_Vector = "Vec"
Field_Integer = "Int"

#
# def pytest_namespace():
#     return {
#         "field_vector": Field_Vector,
#         "field_integer": Field_Integer
#     }


def pytest_addoption(parser):
    parser.addoption("--ip", action="store", default=default_host)
    parser.addoption("--handler", action="store", default="GRPC")
    parser.addoption("--port", action="store", default=default_grpc_port)
    parser.addoption("--dim", action="store", default=default_vector_dim)


@pytest.fixture(scope="module")
def host(request):
    ip_ = request.config.getoption("--ip")
    port_ = request.config.getoption("--port")

    return ip_, port_


@pytest.fixture(scope="module")
def handler(request):
    return request.config.getoption("--handler")


@pytest.fixture(scope='module')
def dim(request):
    return request.config.getoption("--dim")


@pytest.fixture(scope="module")
def connect(request, handler):
    ip = request.config.getoption("--ip")
    handler = request.config.getoption("--handler")
    port_default = default_http_port if handler == "HTTP" else default_grpc_port
    port = request.config.getoption("--port", default=port_default)

    client = Milvus(host=ip, port=port, handler=handler)

    def fin():
        try:
            client.close()
        except:
            pass

    request.addfinalizer(fin)
    return client


@pytest.fixture(scope="module")
def args(request):
    ip = request.config.getoption("--ip")
    port = request.config.getoption("--port")
    args = {"ip": ip, "port": port}
    return args


@pytest.fixture(scope="function")
def vcollection(request, connect, dim):
    ori_collection_name = getattr(request.module, "collection_id", "test")
    collection_name = gen_unique_str(ori_collection_name)
    collection_param = {
        "fields": [
            {"name": Field_Vector, "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}},
        ],
        "segment_row_limit": 8192,
        "auto_id": True
    }

    connect.create_collection(collection_name, collection_param)

    def teardown():
        connect.drop_collection(collection_name)

    request.addfinalizer(teardown)

    return collection_name


@pytest.fixture(scope="function")
def vicollection(request, connect, dim):
    ori_collection_name = getattr(request.module, "collection_id", "test")
    collection_name = gen_unique_str(ori_collection_name)
    collection_param = {
        "fields": [
            {"name": Field_Vector, "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}},
        ],
        "segment_row_limit": 8192,
        "auto_id": False
    }

    connect.create_collection(collection_name, collection_param)

    def teardown():
        connect.drop_collection(collection_name)

    request.addfinalizer(teardown)

    return collection_name


@pytest.fixture(scope="function")
def hvcollection(request, connect, dim):
    collection_name = fake.collection_name()
    collection_param = {
        "fields": [
            {"name": Field_Integer, "type": DataType.INT64},
            {"name": Field_Vector, "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}},
        ],
        "segment_row_limit": 8192,
        "auto_id": True
    }

    connect.create_collection(collection_name, collection_param)

    def teardown():
        table_names = connect.list_collections()
        for name in table_names:
            connect.drop_collection(name)

    request.addfinalizer(teardown)

    return collection_name


@pytest.fixture(scope="function")
def hvicollection(request, connect, dim):
    collection_name = fake.collection_name()
    collection_param = {
        "fields": [
            {"name": Field_Integer, "type": DataType.INT64},
            {"name": Field_Vector, "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}},
        ],
        "segment_row_limit": 8192,
        "auto_id": False
    }

    connect.create_collection(collection_name, collection_param)

    def teardown():
        table_names = connect.list_collections()
        for name in table_names:
            connect.drop_collection(name)

    request.addfinalizer(teardown)

    return collection_name


@pytest.fixture(scope='function')
def vrecords(request, connect, vcollection, dim):

    vectors = records_factory(dim, 10000)

    entities = [
        {"name": Field_Vector, "values": vectors, "type": DataType.FLOAT_VECTOR}
    ]
    connect.insert(vcollection, entities)

    return vcollection


@pytest.fixture(scope='function')
def virecords(request, connect, vicollection, dim):

    vectors = records_factory(dim, 10000)

    entities = [
        {"name": Field_Vector, "values": vectors, "type": DataType.FLOAT_VECTOR}
    ]

    ids = [i for i in range(10000)]

    connect.insert(vcollection, entities, ids)

    return vcollection


@pytest.fixture(scope='function')
def ivrecords(request, connect, hvcollection, dim):

    vectors = records_factory(dim, 10000)
    integers = integer_factory(10000)

    entities = [
        {"name": Field_Integer, "values": integers, "type": DataType.INT64},
        {"name": Field_Vector, "values": vectors, "type": DataType.FLOAT_VECTOR}
    ]
    connect.insert(vcollection, entities)

    return hvcollection


@pytest.fixture(scope='function')
def ivirecords(request, connect, hvicollection, dim):

    vectors = records_factory(dim, 10000)
    integers = integer_factory(10000)

    entities = [
        {"name": Field_Integer, "values": integers, "type": DataType.INT64},
        {"name": Field_Vector, "values": vectors, "type": DataType.FLOAT_VECTOR}
    ]
    ids = [i for i in range(10000)]
    connect.insert(vcollection, entities, ids)

    return hvcollection
