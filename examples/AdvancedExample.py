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
LOG_FORMAT = '%(message)s\n'
LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
# log_handler.setLevel(logging.DEBUG)


_HOST = 'localhost'
_PORT = '19530'
RETRY_TIMES = 3
NORMAL_TIMES = 10

DIMENSION = 16
NUMBER = 200000
TOPK = 1
TABLE_NAME = 'DEMO'


# Decorator to calculate function running time
def time_it(func):
    def inner(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        return ret, end - start
    return inner


class ConnectionHandler(object):
    """Handler connection with the server

    reconnection and connection errors are properly handled

    """

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

    def reconnect(self):
        self._client.connect(self.host, self.port, timeout=1000)

    def _connect(self):
        self._client = Milvus()
        self._client.connect(host=self.host, port=self.port, timeout=1000)

    def connect(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            self._connect()
            while self.can_retry:
                try:

                    return func(*args, **kwargs)

                except NotConnectError as e:
                    LOGGER.error(e)
                    self._retry_times += 1
                    if self.can_retry:
                        LOGGER.warning('Reconnecting to {} .. {}'.
                                       format(_HOST + ':' + _PORT, self._retry_times))
                        self.reconnect()
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
def create_table():
    # Create table 'demo_table' if it doesn't exist.

    if not api.client.has_table(TABLE_NAME):
        LOGGER.info('Creating table `{}` ...'.format(TABLE_NAME))
        param = {
            'table_name': TABLE_NAME,
            'dimension': DIMENSION,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }

        status = api.client.create_table(param)
        if status.OK():
            LOGGER.info('Table `{}` successfully created!'.format(TABLE_NAME))
        else:
            LOGGER.error(status.message)
    else:
        LOGGER.info('Table already existed!')


@api.connect
def describe_table():
    # Get schema of table demo_table
    status, schema = api.client.describe_table(TABLE_NAME)
    if status.OK():
        LOGGER.info('Describing table `{}` ... :'.format(TABLE_NAME))
        print('    {}'.format(schema), end='\n\n')
    else:
        LOGGER.error(status.message)


@api.connect
def add_vectors():

    # Generating 200000 vectors
    LOGGER.info('Generating {} {}-dimension vectors ...'.format(NUMBER, DIMENSION))
    records = [[random.random() for _ in range(DIMENSION)] for _ in range(NUMBER)]

    # Add vectors
    status, ids = api.client.add_vectors(table_name=TABLE_NAME, records=records)
    if status.OK():
        LOGGER.info('Adding vectors to table `{}` ...'.format(TABLE_NAME))

    # Sleeping to wait for data persisting
    LOGGER.info('Sleeping for 6s ... ')
    time.sleep(6)


@time_it
@api.connect
def search_vectors():

    # Search 1 vector
    q_records = [[random.random() for _ in range(DIMENSION)]]

    param = {
        'table_name': TABLE_NAME,
        'query_records': q_records,
        'top_k': TOPK,
    }

    # Search vector
    status, results = api.client.search_vectors(**param)
    if status.OK():
        LOGGER.info('Searching {} nearest vectors from table `{}` ... '.format(TOPK, TABLE_NAME))
        pprint(results)
        print('\n')
    else:
        LOGGER.error(status.message)

    # Get table row counts
    status, counts = api.client.get_table_row_count(TABLE_NAME)
    if status.OK():
        LOGGER.info('Getting table `{}`\'s row counts ...'.format(TABLE_NAME))
        LOGGER.info('    Table row counts: {}'.format(counts))
    else:
        LOGGER.error(status.message)


@api.connect
def delete_table():
    # Delete table
    status = api.client.delete_table(TABLE_NAME)
    if status.OK():
        LOGGER.info('Deleting table `{}` ... '.format(TABLE_NAME))
    else:
        LOGGER.error(status.message)


if __name__ == 'AdvancedExample':

    # Get version of client and server
    version()

    # Create table
    create_table()

    # Describe table
    describe_table()

    # Add vectors
    add_vectors()

    # Search vectors
    _, b = search_vectors()

    # Delete table
    delete_table()

    # Print time of searching vectors
    LOGGER.info('Time of search 1 vector of top{} in table `{}`: {}s'.format(TOPK, TABLE_NAME, b))
