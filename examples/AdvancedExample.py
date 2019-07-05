from milvus import Milvus, IndexType
from milvus.client.Exceptions import *
from contextlib import contextmanager
from functools import wraps
from pprint import pprint
import sys
import time
import random

__name__ = 'AdvancedExample'

_HOST = 'localhost'
_PORT = '19530'
RETRY_TIMES = 3
NORMAL_TIMES = 10

DIMENSION = 16
NUMBER = 200000
TOPK = 10
TABLE_NAME = 'DEMO'

HEADER = "            id              |       L2 distance        "
FORMAT = "    {}     |    {}    "


def generate_vectors():
    # Generating 200000 vectors
    print('Generating {} {}-dimension vectors ...\n'.format(NUMBER, DIMENSION))
    records = [[random.random() for _ in range(DIMENSION)] for _ in range(NUMBER)]
    return records


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
        return self._connect(10000)

    @property
    def can_retry(self):
        if self._normal_times >= NORMAL_TIMES:
            self._retry_times = self._retry_times - 1 if self._retry_times > 0 else 0
            self._normal_times -= NORMAL_TIMES
        return self._retry_times <= RETRY_TIMES

    def _connect(self, timeout=1000):
        client = Milvus()
        client.connect(self.host, self.port, timeout=timeout)
        return client

    def connect(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            while self.can_retry:
                try:
                    return func(*args, **kwargs)
                except NotConnectError:
                    self._retry_times += 1
                    if self.can_retry:
                        print('Reconnecting to {} .. {}'.
                              format(_HOST + ':' + _PORT, self._retry_times))
                        continue
                    else:
                        sys.exit(1)
        return inner


api = ConnectionHandler(host=_HOST, port=_PORT)


@api.connect
def version():

    # get client version
    version = api.client.client_version()
    print('Client version: {}\n'.format(version))

    # get server version
    status, version = api.client.server_version()
    if status.OK():
        print('Server version: {}\n'.format(version))
    else:
        print(status.message)


@api.connect
def create_table():

    # Create table 'demo_table' if it doesn't exist.
    if not api.client.has_table(TABLE_NAME):
        print('Creating table `{}` ...\n'.format(TABLE_NAME))
        param = {
            'table_name': TABLE_NAME,
            'dimension': DIMENSION,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }

        status = api.client.create_table(param)
        if status.OK():
            print('Table `{}` successfully created!\n'.format(TABLE_NAME))
        else:
            print(status.message)
    else:
        print('Table already existed!\n')


@api.connect
def describe_table():

    # Get schema of table demo_table
    status, schema = api.client.describe_table(TABLE_NAME)
    if status.OK():
        print('Describing table `{}` ... :\n'.format(TABLE_NAME))
        print('    {}'.format(schema), end='\n\n')
    else:
        print(status.message)


@api.connect
def add_vectors(records):

    # Add vectors
    status, ids = api.client.add_vectors(table_name=TABLE_NAME, records=records)
    if status.OK():
        print('Adding vectors to table `{}` ...\n'.format(TABLE_NAME))

    # Sleeping to wait for data persisting
    print('Sleeping for 6s ... \n')
    time.sleep(6)

    # Get table row counts
    status, counts = api.client.get_table_row_count(TABLE_NAME)
    if status.OK():
        print('Getting table `{}`\'s row counts ...\n'.format(TABLE_NAME))
        print('    Table row counts: {}\n'.format(counts))
    else:
        print(status.message)


@api.connect
def search_vectors(q_records):

    # Search 1 vector
    param = {
        'table_name': TABLE_NAME,
        'query_records': q_records,
        'top_k': TOPK,
    }

    # Search vector
    status, results = api.client.search_vectors(**param)
    if status.OK():

        print('Get a vector from table `{}`. '
              'Searching its {} nearest vectors from table `{}` ... \n'.
              format(TABLE_NAME, TOPK, TABLE_NAME))
        print(HEADER)
        print('-' * 28 + '+' + '-'*27)
        for result in results:
            for qr in result:
                print(FORMAT.format(qr.id, qr.distance))
        print('')
    else:
        print(status.message)


@api.connect
def delete_table():

    # Delete table
    status = api.client.delete_table(TABLE_NAME)
    if status.OK():
        print('Deleting table `{}` ... \n'.format(TABLE_NAME))
    else:
        print(status.message)


if __name__ == 'AdvancedExample':

    # Get version of client and server
    version()

    # Create table
    create_table()

    # Describe table
    describe_table()

    # Generate vectors
    records = generate_vectors()

    # Add vectors
    add_vectors(records)

    # Search by the first vector added
    q_record = [records[0]]

    # Search vectors
    search_vectors(q_record)

    # Delete table
    delete_table()
