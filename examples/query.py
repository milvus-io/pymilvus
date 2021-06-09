from pymilvus import Milvus, DataType
import random
from pprint import pprint

if __name__ == "__main__":
    c = Milvus("localhost", "19530")

    collection_name = f"test_{random.randint(10000, 99999)}"

    c.create_collection(collection_name, {"fields": [
        {
            "name": "f1",
            "type": DataType.FLOAT_VECTOR,
            "metric_type": "L2",
            "params": {"dim": 4},
            "indexes": [{"metric_type": "L2"}]
        },
        {
            "name": "age",
            "type": DataType.FLOAT,
        },
        {
            "name": "id",
            "type": DataType.INT64,
            "is_primary": True,
        }
    ],
        "auto_id": False,
    }, orm=True)

    assert c.has_collection(collection_name)

    c.insert(collection_name, [
        {"name": "f1", "type": DataType.FLOAT_VECTOR, "values": [[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8]]},
        {"name": "age", "type": DataType.FLOAT, "values": [3.45, 8.9]},
        {"name": "id", "type": DataType.INT64, "values": [1, 2]}
    ], orm=True)

    c.flush([collection_name])

    c.load_collection(collection_name)

    pprint(c.query(collection_name, f"id in [ 1, 2 ]"))
