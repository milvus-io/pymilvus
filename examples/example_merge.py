import random
import time
from milvus import *

if __name__ == '__main__':
    client = Milvus()
    dim = 128
    collection_name = "test_merge2_"

    collection_param = {
        "fields": [
            {"field": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, "metric_type": "L2"}}
        ],
        "segment_size": 100
    }

    client.create_collection(collection_name, collection_param)
    ids_list = list()
    for _ in range(2):
        vectors = [[random.random() for _ in range(dim)] for _ in range(10000)]
        entities = [
            {"field": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR}
        ]
        ids = client.insert(collection_name, entities)
        client.flush()
        ids_list.append(ids)

    for i in range(2):
        del_ids = ids_list[i][:5000]
        client.delete_entity_by_id(collection_name, del_ids)
    client.flush()

    time.sleep(2)
    ## trigger merge operation
    client.flush([collection_name])
    stats = client.get_collection_stats(collection_name)
    seg_id = stats["partitions"][0]["segments"][0]["id"]
    ids = client.list_id_in_segment(collection_name, seg_id)
    time.sleep(1)
