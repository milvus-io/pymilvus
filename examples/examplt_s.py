import sys
import numpy as np
import datetime

sys.path.append(".")
from milvus import Milvus, IndexType

if __name__ == '__main__':
    client = Milvus()
    client.connect(host="localhost", port="19531")

    table_name = "table_name_test"

    status, ok = client.has_table(table_name)
    if not ok:
        client.create_table(
            {
                'table_name': table_name,
                'dimension': 128
            }
        )

        print("Create table Done, Start add vectors")
        records = [np.random.rand(128).astype(np.float32).tolist() for i in range(1000000)]
        # records = [[random.random() for j in range(128)] for i in range(1000000)]
        client.add_vectors(table_name, records=records, timeout=3000)

    query_records = [np.random.rand(128).astype(np.float32).tolist() for i in range(1000)]
    print("[{}] <Interface> Start search ....".format(datetime.datetime.now()))
    client.search_vectors(table_name, top_k=1000, nprobe=16, query_records=query_records)
    print("[{}] <Interface> Search done.".format(datetime.datetime.now()))
