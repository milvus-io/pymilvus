import random

from milvus import *


_HOST = '127.0.0.1'
_PORT = '19530'  # default value


if __name__ == '__main__':
    milvus = Milvus(_HOST, _PORT)

    num = random.randint(1, 100)
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collection_{}'.format(num)

    collection_param = {
        "fields": [
            {"field": "A", "type": DataType.INT32},
            {"field": "B", "type": DataType.INT32},
            {"field": "C", "type": DataType.INT64},
            {"field": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}
        ],
        "segment_size": 100
    }
    milvus.create_collection(collection_name, collection_param)

    milvus.create_index(collection_name, "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}})
    info = milvus.get_collection_info(collection_name)
    print(info)
    milvus.drop_collection(collection_name)

