import random

from pymilvus import (
    MilvusClient,
    FieldSchema, CollectionSchema, DataType,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search


_HOST = '127.0.0.1'
_PORT = '19530'
_URI = f"https://{_HOST}:{_PORT}"

# Const names
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3


def main():
    # create a connection
    print(f"\nCreate connection...")
    milvus_client = MilvusClient(uri=_URI,
                            secure=True,
                            server_pem_path="cert/server.pem",
                            server_name='localhost')
    print(f"\nList connection:")
    print(milvus_client._get_connection())

    # drop collection if the collection exists
    if milvus_client.has_collection(_COLLECTION_NAME):
        milvus_client.drop_collection(_COLLECTION_NAME)

    # create collection
    field1 = FieldSchema(name=_ID_FIELD_NAME, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=_VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    milvus_client.create_collection(collection_name=_COLLECTION_NAME,schema=schema)
    milvus_client.describe_collection(collection_name=_COLLECTION_NAME)

    print("\ncollection created:", _COLLECTION_NAME)

    # show collections
    print("\nlist collections:")
    print(milvus_client.list_collections())

    # insert 10000 vectors with 128 dimension
    data_dict = []
    for i in range(10000):
        entity = {
        "id_field": i+1,  # Assuming id_field is the _COLLECTION_NAME of the field corresponding to the ID
        "float_vector_field": [random.random() for _ in range(_DIM)]
        }
        data_dict.append(entity)
    insert_result = milvus_client.insert(collection_name=_COLLECTION_NAME,data=data_dict)

    # get the number of entities
    print(f"\nThe number of entity: {insert_result['insert_count']}")

    # create index
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name=_VECTOR_FIELD_NAME, 
        index_type=_INDEX_TYPE,
        metric_type=_METRIC_TYPE,
        params={"nlist": _NLIST}
    )

    milvus_client.create_index(
        collection_name=_COLLECTION_NAME,
        index_params=index_params
    )
    print("\nCreated index")

    # load data to memory
    milvus_client.load_collection(_COLLECTION_NAME)
    vector = data_dict[1]
    vectors = [vector["float_vector_field"]]

    # search
    search_param = {
        "anns_field": _VECTOR_FIELD_NAME,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "expr": f"{_ID_FIELD_NAME} > 0"}
    results = milvus_client.search(collection_name=_COLLECTION_NAME,data=vectors,limit= _TOPK,search_params=search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))

    # release memory
    milvus_client.release_collection(_COLLECTION_NAME)

    # drop collection index
    milvus_client.drop_index(_COLLECTION_NAME,index_name=_VECTOR_FIELD_NAME)
    print("\nDrop index sucessfully")

    # drop collection
    milvus_client.drop_collection(_COLLECTION_NAME)
    print("\nDrop collection: {}".format(_COLLECTION_NAME))


if __name__ == '__main__':
    main()
