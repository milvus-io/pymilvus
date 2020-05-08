import random
from milvus import *

if __name__ == '__main__':
    client = Milvus(host="localhost", port=19121, handler="HTTP")
    collection = "test_search_by_id"
    status = client.create_collection({"collection_name": collection, "dimension": 128})
    # assert status.OK()

    vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
    status, ids = client.insert(collection, vectors)
    assert status.OK()

    status = client.flush([collection])
    assert status.OK()

    search_param = {
        "nprobe": 10
    }
    status, results = client.search_by_ids(collection, ids[0: 10], 10, params=search_param)
    print(status), print(results)
    assert status.OK(), status.message
