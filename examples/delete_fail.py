# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.

import copy
import datetime
import sys

sys.path.append(".")
import random
import time
from milvus import Milvus, DataType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
# _HOST = '192.168.1.113'
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 128  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    milvus = Milvus(_HOST, _PORT)

    num = random.randint(100000, 100000 + 2)
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collection_{}'.format(num)

    collection_param = {
        "fields": [
            {"field": "A", "type": DataType.INT32},
            {"field": "B", "type": DataType.INT32},
            {"field": "C", "type": DataType.INT64},
            {"field": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, "metric_type": "L2"}}
        ],
        "segment_size": 100
    }
    milvus.create_collection(collection_name, collection_param)

    milvus.create_partition(collection_name, "p_01")
    milvus.create_partition(collection_name, "p_02")
    # milvus.create_partition(collection_name, "p_01", timeout=1800)
    # pars = milvus.list_partitions(collection_name)
    # ok = milvus.has_partition(collection_name, "p_01", timeout=1800)
    # assert ok
    # ok = milvus.has_partition(collection_name, "p_02")
    # assert not ok
    # for p in pars:
    #     if p == "_default":
    #         continue
    #     milvus.drop_partition(collection_name, p)

    # milvus.drop_collection(collection_name)
    # sys.exit(0)

    A_list = [random.randint(0, 255) for _ in range(num)]
    vec = [[random.random() for _ in range(128)] for _ in range(num)]
    hybrid_entities = [
        {"field": "A", "values": A_list, "type": DataType.INT32},
        {"field": "B", "values": A_list, "type": DataType.INT32},
        {"field": "C", "values": A_list, "type": DataType.INT64},
        {"field": "Vec", "values": vec, "type": DataType.FLOAT_VECTOR}
    ]
    hybrid_entities2 = copy.deepcopy(hybrid_entities)
    ids0 = milvus.insert(collection_name, hybrid_entities, partition_tag="p_01")
    ids1 = milvus.insert(collection_name, hybrid_entities2, partition_tag="p_02")
    milvus.flush([collection_name])
    print("Flush ... ")

    milvus.delete_entity_by_id(collection_name, [ids0[0], ids1[0]])
    print("After delete")  # , flush async")
    milvus.flush([collection_name])
    # print("flush async")
    # time.sleep(3)
    # t0 = time.time()
    # milvus.drop_partition(collection_name, "p_01")
    # print("Drop partition success, waiting for delete done ... | ", time.time() - t0, " s")
    # ff.result()
    # sys.exit(0)

    print("Create index ......")
    milvus.create_index(collection_name, "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})
    print("Create index done.")

    count = milvus.count_entities(collection_name)
    assert count == num * 2 - 2

    info = milvus.get_collection_info(collection_name)
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {
                        "A": [1, 2, 5]
                    }
                },
                {
                    "range": {
                        "B": {"GT": 1, "LT": 100}
                    }
                },
                {
                    "vector": {
                        "Vec": {
                            "topk": 10, "query": vec[: 1], "params": {"nprobe": 10}
                        }
                    }
                }
            ],
        },
    }

    # print("Start searach ..", flush=True)
    # results = milvus.search(collection_name, query_hybrid)
    # print(results)
    #
    # for r in list(results):
    #     print("ids", r.ids)
    #     print("distances", r.distances)

    results = milvus.search(collection_name, query_hybrid, fields=["B"])
    for r in list(results):
        print("ids", r.ids)
        print("distances", r.distances)
        for rr in r:
            print(rr.entity.get("B"))

    # for result in results:
    #     for r in result:
    #         print(f"{r}")

    # itertor entity id
    # for result in results:
    #     for r in result:
    #         # get distance
    #         dis = r.distance
    #         id_ = r.id
    #         # obtain all field name
    #         fields = r.entity.fields
    #         for f in fields:
    #             # get field value by field name
    #             # fv = r.entity.
    #             fv = r.entity.value_of_field(f)
    #             print(fv)

    milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
