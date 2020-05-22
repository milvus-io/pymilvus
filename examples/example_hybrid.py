# This program demos how to connect to Milvus vector database, 
# create a vector collection,
# insert 10 vectors, 
# and execute a vector similarity search.
import datetime
import sys

sys.path.append(".")
import random
import threading
import time
from milvus import Milvus, IndexType, MetricType, Status
from milvus import DataType, RangeType

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

    num = 1000
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collection_{}'.format(num)

    collection_fields = [
        {"field_name": "A", "data_type": DataType.INT64},
        {"field_name": "B", "data_type": DataType.INT64},
        {"field_name": "C", "data_type": DataType.INT64},
        {"field_name": "Vec", "dimension": 128, "extra_params": {"index_file_size": 100, "metric_type": MetricType.L2}}
    ]
    status = milvus.create_hybrid_collection(collection_name, collection_fields)
    print(status)

    A_list = [random.randint(0, 255) for _ in range(num)]
    vec = [[random.random() for _ in range(128)] for _ in range(num)]
    hybrid_entities = [
        {"field_name": "A", "field_values": A_list},
        {"field_name": "B", "field_values": A_list},
        {"field_name": "C", "field_values": A_list},
    ]
    vector_entities = [
        {"field_name": "Vec", "field_values": vec}
    ]
    status, ids = milvus.insert_hybrid(collection_name, hybrid_entities, vector_entities)
    print("Insert done. {}".format(status))
    status = milvus.flush([collection_name])
    print("Flush: {}".format(status))

    query_hybrid = {
        "bool": {
            "must": [
                {
                    "term": {
                        "A": {"values": [1, 2, 5]}
                    }
                },
                {
                    "range": {
                        "B": {"ranges": {RangeType.GT: 1, RangeType.LT: 100}}
                    }
                },
                {
                    "vector": {
                        "Vec": {"topk": 10, "query": vec[: 1], "params": {"nprobe": 10}}
                    }
                }
            ],
        },
        # "fields": []
    }
    status, results = milvus.search_hybrid(collection_name, query_hybrid)
    print(status)

    field_names = results.field_names()
    print("Field\t\tValue")
    for r in results:
        for rr in r:
            for field in field_names:
                print(f"{field}\t\t{rr.get(field)}")

    status, entities = milvus.get_hybrid_entity_by_id(collection_name, ids[:2])


if __name__ == '__main__':
    main()
