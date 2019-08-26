import sys
import random
import time
import concurrent.futures

sys.path.append('.')
from factorys import time_it

dimension = 512
number = 100000
table_name = 'multi_task'


def add_vector_task(milvus, vector):
    status, ids = milvus.add_vectors(table_name=table_name, records=vector)

    assert status.OK(), "add vectors failed"
    assert len(ids) == len(vector)


@time_it
def grpc_thread_pool_add_vector(milvus, pool_size, vectors):
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(pool_size):
            executor.submit(add_vector_task, milvus, vectors)


def test_grpc_run(gcon):
    gmilvus = gcon

    if gmilvus is None:
        assert False, "Error occurred: connect failure"

    if gmilvus.has_table(table_name):
        gmilvus.delete_table(table_name)
        time.sleep(2)

    table_param = {
        'table_name': table_name,
        'dimension': dimension,
        'index_file_size': 1024
    }

    gmilvus.create_table(table_param)

    for p in (1, 5, 10, 20, 40, 50, 100):
        pool_size = p
    step = number // pool_size

    vectors = [[random.random() for _ in range(dimension)] for _ in range(step)]

    print(f'Thread pool size................... {p}')
    grpc_thread_pool_add_vector(gmilvus, p, vectors)

    time.sleep(1.5)
    _, gcount = gmilvus.get_table_row_count(table_name)
    print(gcount)


# if __name__ == '__main__':
#     grpc_run()
