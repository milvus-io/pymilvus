import sys
sys.path.append('.')
import struct
import random
from milvus.client.GrpcClient import Prepare
from milvus.client.Abstract import QueryResult, TopKQueryResult
from pprint import pprint
from functools import wraps
import time


# TOPK=10000, nq=1000 deserialize costs 6.7s
TOPK = 10000
nq = 1000
DIM = 512
NUM = 100000

def time_it(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()

        a = func(*args, **kwargs)

        dur = time.perf_counter() - start
        print(f"[{func.__name__}]: {dur}")

        return dur
    return inner


def gen_one_binary(topk):
    ids = [random.randrange(10000000, 99999999) for _ in range(topk)]
    distances = [random.random() for _ in range(topk)]
    return struct.pack(str(topk) + 'l', *ids), struct.pack(str(topk) + 'd', *distances)


def gen_nq_binaries(nq, topk):
    return [gen_one_binary(topk) for _ in range(nq)]


@time_it
def deserialize(nums, nq, topk):
    a = TopKQueryResult()

    for topks in nums:
        ids = topks[0]
        distances = topks[1]
        ids = struct.unpack(str(topk) + 'l', ids)
        distances = struct.unpack(str(topk) + 'd', distances)

        qr = [QueryResult(ids[i], distances[i]) for i in range(topk)]
        assert len(qr) == topk

        a.append(qr)
    assert len(a) == nq
    return a


def de_topk_query_result():
    '''Test serialize TopKQueryResult'''
    g = gen_nq_binaries(nq, TOPK)
    result = deserialize(g, nq, TOPK)
    pprint(result[1][1])


def gen_arrays(number, dimension):
    '''gen vectors for given dimension and number'''
    return [[random.random() for _ in range(dimension) ] for _ in range(number)]

@time_it
def serialize_insert_infos_one_thread(arrays):
    a = Prepare.insert_infos('table_name', arrays)


def run_one_thread(number):
    arrays = gen_arrays(number, DIM)
    a = serialize_insert_infos_one_thread(arrays)
    return a

def run_one_thread_multi_times():
    result = []
    for number in range(100000, 1000000, 100000):
        print(f"Number: {number}")
        costs = run_one_thread(number)
        result.append((number, costs))
    print(result)


if __name__ == '__main__':
    run_one_thread(NUM)
