import random
import time
import pytest
from milvus.client.utils import merge_results


@pytest.mark.skip
def test_merge_results(gcon):
    table_name = "merge_test"

    # delete table is exists
    status, ok = gcon.has_table(table_name)
    if ok:
        gcon.drop_table(table_name)

    table = {"table_name": table_name, "dimension": 128, "index_file_size": 100}
    gcon.create_table(table)

    vectors = [[random.random() for _ in range(128)] for _ in range(1000 * 1000)]

    gcon.insert(table_name, vectors)
    time.sleep(5)

    query_1_vectors = vectors[: 5]

    results_list = []
    for i in range(2000):
        status, results = gcon.search_in_files(table_name, top_k=5, nprobe=5, query_records=query_1_vectors,
                                               file_ids=[i])
        if status.OK():
            results_list.append(results.raw)

    topk_merge_result = merge_results(results_list, 5)

    status, result = gcon.search(table_name, top_k=5, nprobe=5, query_records=query_1_vectors)

    for merge_ids, r_ids in zip(topk_merge_result.id_array, result.id_array):
        for merge_id, r_id in zip(merge_ids, r_ids):
            assert merge_id == r_id

    gcon.drop_table(table_name)
