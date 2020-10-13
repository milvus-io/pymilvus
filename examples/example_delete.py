# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.
import datetime
import sys

sys.path.append(".")
import random
import time
from milvus import Milvus, DataType
from milvus import utils

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

    # num = random.randint(1, 100000)
    num = 100000
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collections_{}'.format(num)
    if milvus.has_collection(collection_name):
        milvus.drop_collection(collection_name)

    collection_param = {
        "fields": [
            {"name": "A", "type": DataType.INT32},
            {"name": "B", "type": DataType.INT32},
            {"name": "C", "type": DataType.INT64},
            {"name": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, "metric_type": "L2"}}
        ],
        "segment_size": 100
    }
    milvus.create_collection(collection_name, collection_param)

    milvus.compact(collection_name)

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
        {"name": "A", "values": A_list, "type": DataType.INT32},
        {"name": "B", "values": A_list, "type": DataType.INT32},
        {"name": "C", "values": A_list, "type": DataType.INT64},
        {"name": "Vec", "values": vec, "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}
    ]

    for slice_e in utils.entities_slice(hybrid_entities):
        ids = milvus.insert(collection_name, slice_e)
    milvus.flush([collection_name])
    print("Flush ... ")
    # time.sleep(3)
    count = milvus.count_entities(collection_name)

    milvus.delete_entity_by_id(collection_name, ids[:1])
    milvus.flush([collection_name])
    print("Get entity be id start ...... ")
    entities = milvus.get_entity_by_id(collection_name, ids[:1])
    et = entities.dict()
    milvus.delete_entity_by_id(collection_name, ids[1:2])
    milvus.flush([collection_name])

    print("Create index ......")
    milvus.create_index(collection_name, "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})
    print("Create index done.")

    info = milvus.get_collection_info(collection_name)
    print(info)
    stats = milvus.get_collection_stats(collection_name)
    print("\nstats\n")
    print(stats)
    query_hybrid = \
    {
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
                            "topk": 10, "query": vec[: 10000], "params": {"nprobe": 10}
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

    t0 = time.time()
    count = 0
    results = milvus.search(collection_name, query_hybrid, fields=["B"])
    for r in list(results):
        # print("ids", r.ids)
        # print("distances", r.distances)
        for rr in r:
            count += 1
            # print(rr.entity.get("B"))

    print("Search cost {} s".format(time.time() - t0))

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

# {
#     "fields": [
#         {"field": "A", "type": DataType.INT32},
#         {"field": "B", "type": DataType.INT32},
#         {"field": "C", "type": DataType.INT64},
#         {"field": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, "metric_type": "L2"}}
#     ],
#     "segment_size": 100
# }

