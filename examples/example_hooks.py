# This program is a simple demo on how to use serch hook in pymilvus.
# Customize a search hook class to count search time
import random
import datetime
import time
from milvus import Milvus
from milvus.client.hooks import BaseSearchHook


# Define a subclass of `BaseSearchHook` which print time stamp
# when client call server and call done.
class CustomizedSearchHook(BaseSearchHook):

    def __init__(self):
        self._start_stamp = None
        self._end_stamp = None

    def pre_search(self, *args, **kwargs):
        # record time when call on server start.
        self._start_stamp = datetime.datetime.now()
        print("[{}] <hook> Start send request to server.".format(self._start_stamp))

    def aft_search(self, *args, **kwargs):
        # record time when call on server done.
        self._end_stamp = datetime.datetime.now()
        print("[{}] <hook> Server done.".format(self._end_stamp))


if __name__ == '__main__':
    # Server host and port, you may need to change accordingly
    _ADDRESS = {
        "host": "127.0.0.1",
        "port": "19530"
    }

    # Table name
    table_name = "demo_hooks"
    # Dimension of vectors
    dim = 128

    table_param = {
        "table_name": table_name,
        "dimension": dim
    }

    # generate 10000 vectors with dimension of 128 randomly
    vectors = [[random.random() for _ in range(dim)] for _ in range(10000)]

    # select part of vectors as query vectors
    query_vectors = vectors[5: 10]

    # Start request to server.
    # You can invoke without `with` here to interact with server:
    #
    #     client = Milvus()
    #     client.connect(**_ADDRESS)
    #             ......
    #     client.disconnect()
    #
    with Milvus(**_ADDRESS) as client:
        # Set search hook
        client.set_hook(search=CustomizedSearchHook())

        # Create table
        client.create_table(table_param)

        # Insert vectors into table
        client.insert(table_name, vectors)

        time.sleep(5)

        # Search approximate vectors in table `demo_hooks`,
        # you can find call information in console output
        client.search(table_name, 10, 10, query_vectors)

        # delete table
        client.drop_table(table_name)
