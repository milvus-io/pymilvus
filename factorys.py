# STL imports
import random
import string
import time
import datetime
import random
import struct
import sys
from functools import wraps

sys.path.append('.')
# Third party imports
import numpy as np
import faker
from faker.providers import BaseProvider

# local application imports
from milvus.client.Abstract import IndexType

# grpc
from milvus.client.GrpcClient import Prepare as gPrepare
from milvus.grpc_gen import milvus_pb2


def gen_vectors(num, dim):
    return [[random.random() for _ in range(dim)] for _ in range(num)]


def gen_single_vector(dim):
    return [[random.random() for _ in range(dim)]]


def gen_vector(nb, d, seed=np.random.RandomState(1234)):
    xb = seed.rand(nb, d).astype("float32")
    return xb.tolist()


def gen_unique_str(str=None):
    prefix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    return prefix if str is None else str + "_" + prefix


def get_current_day():
    return time.strftime('%Y-%m-%d', time.localtime())


def get_last_day(day):
    tmp = datetime.datetime.now() - datetime.timedelta(days=day)
    return tmp.strftime('%Y-%m-%d')


def get_next_day(day):
    tmp = datetime.datetime.now() + datetime.timedelta(days=day)
    return tmp.strftime('%Y-%m-%d')


def gen_long_str(num):
    string = ''
    for _ in range(num):
        char = random.choice('tomorrow')
        string += char


def gen_one_binary(topk):
    ids = [random.randrange(10000000, 99999999) for _ in range(topk)]
    distances = [random.random() for _ in range(topk)]
    return milvus_pb2.TopKQueryResult(struct.pack(str(topk) + 'l', *ids), struct.pack(str(topk) + 'd', *distances))


def gen_nq_binaries(nq, topk):
    return [gen_one_binary(topk) for _ in range(nq)]


def fake_query_bin_result(nq, topk):
    return gen_nq_binaries(nq, topk)


class FakerProvider(BaseProvider):

    def table_name(self):
        return 'table_name' + str(random.randint(1000, 9999))

    def name(self):
        return 'name' + str(random.randint(1000, 9999))

    def dim(self):
        return random.randint(0, 999)


fake = faker.Faker()
fake.add_provider(FakerProvider)


def range_factory():
    param = {
        'start_date': '2019-06-25',
        'end_date': '2019-10-10'
    }
    return gPrepare.range(**param)


def ranges_factory():
    return [range_factory() for _ in range(5)]


def table_schema_factory():
    param = {
        'table_name': fake.table_name(),
        'dimension': random.randint(1, 999),
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    }
    return param


def records_factory(dimension, nq):
    return [[random.random() for _ in range(dimension)] for _ in range(nq)]


def query_ranges_factory():
    param = [('2019-06-25', '2100-1-25')]
    return gPrepare.ranges(param)


def time_it(func):
    @wraps(func)
    def inner(*args, **kwrgs):
        pref = time.perf_counter()
        result = func(*args, **kwrgs)
        delt = time.perf_counter() - pref
        print(f"[{func.__name__}][{delt:.4}s]")
        return result

    return inner
