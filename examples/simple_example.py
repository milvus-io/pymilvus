import random
import pymilvus.orm.pymilvus_orm as orm
from pymilvus import DataType

random.seed(1234567890)


def main():
   
    orm.connections.connect("default", host = "127.0.0.1", port= 19530)

    try:
        collection = orm.collection.Collection(name="simple_example")
        collection.drop()
    except:
        pass

    dim = 128

    id_field = orm.schema.FieldSchema(name="ids_field", dtype=DataType.INT64, description="primary ids", auto_id=True, is_primary=True)
    vector_field = orm.schema.FieldSchema(name="vectors_field", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vectors")
    schema = orm.schema.CollectionSchema(fields =[id_field, vector_field], description="collection of vectors")
    
    collection = orm.collection.Collection(name="simple_example", schema=schema)

    store_vectors = [[random.random() for _ in range(dim)] for _ in range(100000)]

    collection.insert([store_vectors])

    print("Number of vectors: " + str(collection.num_entities))

    index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {"nlist": 512}}

    orm.index.Index(collection, "vectors_field", index_params)

    collection.load()

    search_vectors = [[random.random() for _ in range(dim)] for _ in range(10)]

    search_params = {"nprobe": 128}

    full_params = {"metric_type": "L2", "params": search_params}

    results = collection.search(data=search_vectors, anns_field="vectors_field", param=full_params, limit=3)

    for query_res in results[0]:
        res_id = query_res.id
        res_dist = query_res.distance
        print("id: " + str(res_id), "dist: " + str(res_dist))

    collection.drop()

if __name__ == "__main__":
    main()








    


