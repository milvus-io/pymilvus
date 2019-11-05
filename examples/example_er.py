import sys
sys.path.append(".")
import faker
import numpy as np

from milvus import Milvus, IndexType, MetricType

fake = faker.Faker()

if __name__ == '__main__':
    client = Milvus()
    status = client.connect()
    if status.OK():
        print("Connect OK")

    table_name_list = []

    for i in range(400):
        table_name = fake.word()
        param = {
            'table_name': table_name,
            'dimension': 128,
            'index_file_size': 10
        }
        status = client.create_table(table_name)
        if not status.OK():
            print("Create table {} failed.".format(table_name))


