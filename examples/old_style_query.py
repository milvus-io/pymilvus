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
            "auto_id": True,
            "is_primary": True,
        }
    ],
    }, orm=True)

    assert c.has_collection(collection_name)

    ids = c.insert(collection_name, [
        {"name": "f1", "type": DataType.FLOAT_VECTOR, "values": [[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8]]},
        {"name": "age", "type": DataType.FLOAT, "values": [3.45, 8.9]}
    ], orm=True)

    c.flush([collection_name])

    c.load_collection(collection_name)

    #############################################################
    search_params = {"metric_type": "L2", "params": {"nprobe": 1}}

    results = c.search_with_expression(collection_name, [[1.1, 2.2, 3.3, 4.4]],
                                       "f1", param=search_params, limit=2, output_fields=["id"])

    print("search results: ", results[0][0].entity, " + ", results[0][1].entity)
    #
    # print("Test entity.get: ", results[0][0].entity.get("age"))
    # print("Test entity.value_of_field: ", results[0][0].entity.value_of_field("age"))
    # print("Test entity.fields: ", results[0][0].entity.fields)
    #############################################################

    ids_expr = ",".join(str(x) for x in ids.primary_keys)

    expr = "id in [ " + ids_expr + " ] "

    print(expr)

    res = c.query(collection_name, expr)
    print(res)
