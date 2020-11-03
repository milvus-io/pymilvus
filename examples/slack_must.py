from milvus import Milvus, DataType

if __name__ == '__main__':
    client = Milvus("127.0.0.1", "19530")
    collection_name = "test_slack_dsl2"
    collection_param = {
        "fields": [
            {"name": "int", "type": DataType.INT64},
            {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}}
        ]
    }

    client.create_collection(collection_name, collection_param)

    dsl = {
        "bool": {
            "must": [
                {
                    "must": [
                        {
                            "must_not": [
                                {
                                    "term": {
                                        "int": [1]
                                    }
                                }
                            ]
                        }
                    ]
                },
                {
                    "must": [
                        {
                            "vector": {
                                "vec": {
                                    "topk": 10,
                                    "query": [[0.1, 0.2]],
                                    "metric_type": "L2"
                                }
                            }
                        }
                    ]
                }
            ]
        }
    }

    client.search(collection_name, dsl)