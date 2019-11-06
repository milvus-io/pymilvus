import sys
import numpy as np
import datetime

sys.path.append(".")
from milvus import Milvus, IndexType

sift_1b_base_file = "/exp/hdd1/data/faiss_assets/ANN_SIFT1B/dataset/bigann_base.bvecs"
sift_1b_query_file = "/exp/hdd1/data/faiss_assets/ANN_SIFT1B/dataset/bigann_query.bvecs"
_NB = 1000 * 1000
_NQ = 10000
_Top_K = 1000
_HOST = "192.168.1.113"
# _HOST = "127.0.0.1"
# _PORT = "19531"
_PORT = "19530"


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


if __name__ == '__main__':
    client = Milvus()
    client.connect(host=_HOST, port=_PORT)

    table_name = "table_test"

    status, ok = client.has_table(table_name)
    if not ok:
        client.create_table(
            {
                'table_name': table_name,
                'dimension': 128
            }
        )

        print("Create table Done, Start add vectors")
        # records = [np.random.rand(128).astype(np.float32).tolist() for i in range(1000000)]
        records = sift_xb = mmap_bvecs(sift_1b_base_file)[:_NB].astype(np.float32).tolist()
        # records = [[random.random() for j in range(128)] for i in range(1000000)]
        client.add_vectors(table_name, records=records, timeout=3000)

        print("Start create index")
        status = client.create_index(table_name, {"index_type": IndexType.IVFLAT, "nlist": 4096})
        if status.OK():
            print("Create index OK")

    # query_records = [np.random.rand(128).astype(np.float32).tolist() for i in range(_NQ)]
    query_records = mmap_bvecs(sift_1b_query_file)[:_NQ].astype(np.float32).tolist()

    time_stamp0 = datetime.datetime.now()
    print("[{}] <Interface> Start search ....".format(time_stamp0))
    client.search_vectors(table_name, top_k=_Top_K, nprobe=16, query_records=query_records)
    time_stamp1 = datetime.datetime.now()
    print("[{}] <Interface> Search done.".format(time_stamp1))
    time_r = time_stamp1 - time_stamp0
    print("Search interface cost {} ms".format(time_r.seconds * 1000 + time_r.microseconds // 1000))

