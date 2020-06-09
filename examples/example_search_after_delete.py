import random
import sys

from milvus import *

if __name__ == '__main__':
    collection_name_ = "test_collection_search_after_delete_11"

    client = Milvus(host="192.168.1.57", port=19531)
    client.create_collection({"collection_name": collection_name_, "dimension": 128})

    del_vecs = []
    del_ids = []
    for i in range(5):
        print(f"Insert round {i} ...")
        vectors = [[random.random() for _ in range(128)] for _ in range(100000)]
        del_vecs.append(vectors[0])
        vec_ids = [i * 100000 + j for j in range(100000)]
        del_ids.append(vec_ids[0])
        status, ids = client.insert(collection_name_, vectors, ids=vec_ids)
        if not status.OK():
            print(f"Insert failed: {status}", file=sys.stderr)
            sys.exit(1)

    client.flush([collection_name_])
    print("Create index ... ")
    status = client.create_index(collection_name_, IndexType.IVF_FLAT, params={"nlist": 1024})
    if not status.OK():
        print(f"Create index failed: {status}", file=sys.stderr)
        sys.exit(1)

    print("Search before delete ....")
    status, results = client.search(collection_name_, 5, del_vecs, params={"nprobe": 10})

    print("Delete entity ... ")
    status = client.delete_entity_by_id(collection_name_, id_array=del_ids)
    if not status.OK():
        print(f"Create index failed: {status}", file=sys.stderr)
        sys.exit(1)

    client.flush([collection_name_])
    print("Search after delete ... ")
    status, results = client.search(collection_name_, 5, del_vecs, params={"nprobe": 10})
    if not status.OK():
        print(f"Search failed: {status}", file=sys.stderr)
        sys.exit(1)

    for del_id, result in zip(del_ids, results):
        if result[0].id == del_id:
            print(f"Unexcepted search result. {result[0]}")

    client.drop_collection(collection_name_)
