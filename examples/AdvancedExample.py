from milvus import Milvus, IndexType
from milvus.client.Exceptions import *
from contextlib import contextmanager
from functools import wraps
from pprint import pprint
import sys
import time
import random
import logging

__name__ = 'AdvancedExample'
# logging format
LOG_FORMAT = ('%(name) -10s %(funcName) '
              '-7s: %(message)s')
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

_HOST = '127.0.0.1'
_PORT = '19530'
RETRY_TIMES = 3
NORMAL_TIMES = 10

DIMENSION = 16
TABLE_NAME = 'demo_table'


class ConnectionHandler(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._retry_times = 0
        self._normal_times = 0
        self._client = None

    @property
    def client(self):
        return self._client

    @property
    def can_retry(self):
        if self._normal_times >= NORMAL_TIMES:
            self._retry_times = self._retry_times - 1 if self._retry_times > 0 else 0
            self._normal_times -= NORMAL_TIMES
        return self._retry_times <= RETRY_TIMES

    @contextmanager
    def connect_context(self):

        try:
            self._client = Milvus()
            self._client.connect(_HOST, _PORT)
        except Exception:
            raise RuntimeError("Connection failed")
        yield
        self._client.disconnect()

    def connect(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self.connect_context():
                try:

                    return func(*args, **kwargs)
                except ConnectError as e:
                    LOGGER.error(e)
                    self._retry_times += 1
                    if self.can_retry:
                        LOGGER.warning('Reconnecting to {} .. {}'.
                                       format(_HOST + ':' + _PORT, self._retry_times))
                    else:
                        sys.exit(1)

        return inner


api = ConnectionHandler(host=_HOST, port=_PORT)


@api.connect
def version():
    # get client version
    version = api.client.client_version()
    LOGGER.info('Client version: {}'.format(version))

    # get server version
    status, version = api.client.server_version()
    if status.OK():
        LOGGER.info('Server version: {}'.format(version))
    else:
        LOGGER.error(status.message)


@api.connect
def table():
    # Create table 'demo_table' if it doesn't exist.
    table_name = TABLE_NAME
    dimension = DIMENSION
    if not api.client.has_table(table_name):
        LOGGER.info('Creating table `{}` ...'.format(table_name))
        param = {
            'table_name': table_name,
            'dimension': dimension,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }

        status = api.client.create_table(param)
        if status.OK():
            LOGGER.info('Table `{}` successfully created!'.format(table_name))
        else:
            LOGGER.error(status.message)
    else:
        LOGGER.info('Table `{}` already exists!'.format(table_name))

    # Get schema of table demo_table
    status, schema = api.client.describe_table(table_name)
    if status.OK():
        LOGGER.info('Describing table `{}` ... :'.format(table_name))
        pprint(schema)
    else:
        LOGGER.error(status.message)


@api.connect
def vectors():
    # Generate 20 Fake-vectors
    records = [[random.random() for _ in range(DIMENSION)] for _ in range(20)]

    status, ids = api.client.add_vectors(table_name=TABLE_NAME, records=records)
    if status.OK():
        LOGGER.info('Adding vectors ... ')
        pprint(ids)

    time.sleep(6)

    # Search by the 1st element in records
    q_records = [records[0]]

    param = {
        'table_name': TABLE_NAME,
        'query_records': q_records,
        'top_k': 1,
    }
    status, results = api.client.search_vectors(**param)
    if status.OK():
        LOGGER.info('Searching vectors ... ')
        LOGGER.info('{}'.format(results))
    else:
        LOGGER.error(status.message)

    # Get table row counts
    status, counts = api.client.get_table_row_count(TABLE_NAME)
    if status.OK():
        LOGGER.info('Getting table row counts ... ')
        LOGGER.info('Table row counts: {}'.format(counts))
    else:
        LOGGER.error(status.message)


def time_it(func):
    def inner(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        return ret, end - start
    return inner

import requests
import time
# from PIL import Image
import tensorflow as tf

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np


class VGG16ModelHandler:
    NAME = 'VGG16'
    DIMENSION = 512
    DESCRIPTION = 'System builtin VGG16 model'

    def __init__(self):
        self.handler = VGG16(weights='imagenet',
                             pooling='max',
                             include_top=False)

    # @property
    # def meta(self):
    #     return {
    #         'model_id': self.NAME,
    #         'dimension': self.DIMENSION,:
    #         'description': self.DESCRIPTION
    #     }

    @property
    def name(self):
        return self.NAME

    @property
    def dimension(self):
        return self.DIMENSION

    @time_it
    def predict(self, file):
        LOGGER.error('xx')
        img = image.load_img(file)
        img.resize((224, 224))
        ret = image.img_to_array(img)
        ret = np.expand_dims(ret, axis=0)
        ret = preprocess_input(ret)

        return self.handler.predict(ret)[0]


if __name__ == 'AdvancedExample':
    # version()
    # table()
    # vectors()
    vgg = VGG16ModelHandler()
    vector = vgg.predict('examples/angelatu.jpg')
    print(vectors)
