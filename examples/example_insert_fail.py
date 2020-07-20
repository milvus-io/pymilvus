import random
import sys
import time

from milvus import *

if __name__ == '__main__':
    dimension = 128
    collection_name = "insert_fail"

    insert_vectors = [[random.random() for _ in range(dimension)] for _ in range(100)]

    client = Milvus()

    client.create_collection({"collection_name": collection_name, "dimension": dimension})
    start_time = time.time()
    i = 0
    while time.time() < start_time + 2 * 24 * 3600:
        i = i + 1
        # logger.debug(i)
        # logger.debug("Row count: %d" % milvus_instance.count())
        _, count = client.count_entities(collection_name)
        print("Row count: {}".format(count))
        status, res = client.insert(collection_name, insert_vectors)
        if not status.OK():
            print("Insert failed. {}".format(status))
            sys.exit(1)
        time.sleep(0.1)
