# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.
import datetime
import sys

sys.path.append(".")
import random
import time
from milvus import Milvus, IndexType, MetricType, Status
from milvus import DataType, RangeType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '192.168.1.113'
# _HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 128  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    milvus = Milvus(_HOST, _PORT)

    num = random.randint(1, 100)
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collection_{}'.format(num)

    collection_param = {
        "fields": [
            {"field": "A", "type": DataType.INT64},
            {"field": "B", "type": DataType.INT64},
            {"field": "C", "type": DataType.INT64},
            {"field": "Vec", "type": DataType.VECTOR, "params": {"dimension": 128, "metric_type": "L2"}}
        ],
        "segment_size": 100
    }
    milvus.create_collection(collection_name, collection_param)

    A_list = [random.randint(0, 255) for _ in range(num)]
    vec = [[random.random() for _ in range(128)] for _ in range(num)]
    hybrid_entities = [
        {"field": "A", "values": A_list, "type": DataType.INT64},
        {"field": "B", "values": A_list, "type": DataType.INT64},
        {"field": "C", "values": A_list, "type": DataType.INT64},
        {"field": "Vec", "values": vec, "type": DataType.VECTOR}
    ]

    ids = milvus.insert(collection_name, hybrid_entities)
    milvus.flush([collection_name])
    print("Flush ... ")
    time.sleep(3)

    print("Get entity be id start ...... ")
    entities = milvus.get_entity_by_id(collection_name, ids[:1])
    et = entities.dict()

    print("Create index ......")
    milvus.create_index(collection_name, "Vec", "ivf_flat", {"index_type": "IVF_FLAT", "nlist": 100})
    print("Create index done.")

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
