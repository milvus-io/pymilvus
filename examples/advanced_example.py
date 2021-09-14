import random
import pymilvus.orm.pymilvus_orm as orm
from pymilvus import DataType
from pprint import pprint
random.seed(1234567890)


def main():
   
    orm.connections.connect("default", host = "127.0.0.1", port= 19530)
    orm.connections.connect("collection_con", host = "127.0.0.1", port= 19530)

    try:
        collection = orm.collection.Collection(name="advanced_example")
        collection.drop()
    except:
        pass

    dim = 128

    id_field = orm.schema.FieldSchema(name="ids_field", dtype=DataType.INT64, description="primary ids", auto_id=True, is_primary=True)
    vector_field = orm.schema.FieldSchema(name="vectors_field", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vectors")
    scalar_field = orm.schema.FieldSchema(name="scalar_field", dtype=DataType.INT64, description="numbers")
    int_boolean_field = orm.schema.FieldSchema(name="int_boolean_field", dtype=DataType.INT64, description="0 or 1")

    schema = orm.schema.CollectionSchema(fields =[id_field, vector_field, scalar_field, int_boolean_field], description="collection of vectors")
    
    collection = orm.collection.Collection(name="advanced_example", schema=schema, using="collection_con")

    partitions = []

    for x in range(1,4):
        partition = orm.partition.Partition(collection=collection, name="partition_" + str(x), descrition="partition " + str(x) + " of 3")
        partitions.append(partition)

    print("Partitions before insert: ")
    pprint(collection.partitions)

    for x in partitions:
        amount = 100000
        store_vectors = [[random.random() for _ in range(dim)] for _ in range(amount)]
        store_a = [random.randint(1,amount) for _ in range(amount)]
        store_b = [random.randint(0,1) for _ in range(amount)]

        x.insert([store_vectors, store_a, store_b])

    print("Partitions after insert: ")
    pprint(collection.partitions)


    index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {"nlist": 512}}

    orm.index.Index(collection, "vectors_field", index_params)

    collection.load()

    search_vectors = [[random.random() for _ in range(dim)] for _ in range(10)]

    search_params = {"nprobe": 128}

    full_params = {"metric_type": "L2", "params": search_params}

    partition_names = [x.name for x in partitions]

    search_expression = 'scalar_field < 55252 && int_boolean_field == 1'

    output_fields = ["ids_field"]

    #vvvvvvvvvvvvvvvvv----Will work on RC-6 and above--------------vvvvvvvvvvvvvvvvvvvvv
    print("Rows that fit expression: " +str(len(collection.query(expr=search_expression, partition_names=partition_names))))
    results = collection.search(data=search_vectors, anns_field="vectors_field", param=full_params, limit=3, expr=search_expression, partition_names=partition_names)

    for query_res in results[0]:
        res_id = query_res.id
        res_dist = query_res.distance
        print("id: " + str(res_id), "dist: " + str(res_dist))

    collection.drop()

if __name__ == "__main__":
    main()
