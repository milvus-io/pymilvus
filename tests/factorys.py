# STL imports
import random
import logging
import string
import time
import datetime
import random
import struct
import sys
from functools import wraps

# Third party imports
import numpy as np
import faker
from faker.providers import BaseProvider

logging.getLogger('faker').setLevel(logging.ERROR)

sys.path.append('.')


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


class FakerProvider(BaseProvider):

    def collection_name(self):
        return 'collection_names' + str(random.randint(1000, 9999))

    def name(self):
        return 'name' + str(random.randint(1000, 9999))

    def dim(self):
        return random.randint(0, 999)


fake = faker.Faker()
fake.add_provider(FakerProvider)


def collection_name_factory():
    return fake.collection_name()


def records_factory(dimension, nq):
    return [[random.random() for _ in range(dimension)] for _ in range(nq)]


def binary_records_factory(dim, nq):
    # uint8 values range is [0, 256), so we specify the high range is 256.
    xnb = np.random.randint(256, size=[nq, (dim // 8)], dtype="uint8")
    xb = [bytes(b) for b in xnb]
    return xb


def integer_factory(nq):
    return [random.randint(0, 128) for _ in range(nq)]


def time_it(func):
    @wraps(func)
    def inner(*args, **kwrgs):
        pref = time.perf_counter()
        result = func(*args, **kwrgs)
        delt = time.perf_counter() - pref
        print(f"[{func.__name__}][{delt:.4}s]")
        return result

    return inner
