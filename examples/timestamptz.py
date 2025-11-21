import time
from pymilvus import MilvusClient, DataType, CollectionSchema, IndexType, FieldSchema, milvus_client
import datetime
import pytz

milvus_host = "http://localhost:19530"
collection_name = "timestamptz_test123"


def main():
  client = MilvusClient(uri=milvus_host)

  # create collection with TIMESTAMPTZ field
  if client.has_collection(collection_name):
    client.drop_collection(collection_name)

  schema = client.create_schema()
  schema.add_field("id", DataType.INT64, is_primary=True)
  schema.add_field("tsz", DataType.TIMESTAMPTZ)
  schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
  print("===================alter database timezone===================")
  try:
    client.alter_database_properties("default", {"timezone": "Asia/Shanghai"})
  except Exception as e:
    print(e)
  print(client.describe_database("default"))
  client.create_collection(collection_name, schema=schema, consistency_level="Session")
  index_params = client.prepare_index_params(
    collection_name=collection_name,
    field_name="vec",
    index_type=IndexType.HNSW,
    metric_type="COSINE",
    params={"M": 30, "efConstruction": 200},
  )
  # add timestamptz index of STL_SORT type
  index_params.add_index(field_name="tsz", index_name="tsz_index", index_type="STL_SORT")

  client.create_index(collection_name, index_params)
  print(client.describe_collection(collection_name))
  client.load_collection(collection_name)
  print(f"load state: {client.get_load_state(collection_name)}")

  # insert data with timezone-aware timestamps
  print("===================insert timestamptz===================")
  data_size = 8193
  shanghai_tz = pytz.timezone("Asia/Shanghai")
  data = [
    {
      "id": i + 1,
      "tsz": shanghai_tz.localize(
        datetime.datetime(2025, 1, 1, 0, 0, 0) + datetime.timedelta(days=i)
      ).isoformat(),
      "vec": [float(i) / 10 for i in range(4)],
    }
    for i in range(data_size)
  ]
  client.insert(collection_name, data)
  client.flush(collection_name)
  time.sleep(1) # wait for index creation
  print(client.describe_index(collection_name, "tsz_index"))
  print("===================insert invalid string===================")
  data = [{"id": 114514, "tsz": "should cause an error", "vec": [1.1, 1.2, 1.3]}]
  try:
    client.insert(collection_name, data)
  except Exception as e:
    print(e)

  # query/search data with TIMESTAMPTZ, define timezone in kwargs
  print("====================test query====================")
  results = client.query(
    collection_name,
    filter="id <= 10",
    output_fields=["id", "tsz", "vec"],
    limit=2,
    timezone="America/Havana",
    time_fields="year, month, day, hour, minute, second, microsecond",
  )
  print("\n".join([str(res) for res in results]))

  print("====================test search====================")
  results = client.search(
    collection_name,
    [[0.5, 0.6, 0.7, 0.8]],
    output_fields=["id", "tsz", "vec"],
    limit=10,
    timezone="America/Chicago",
    time_fields="year, month, day, hour, minute, second, microsecond",
  )
  print("\n".join([str(res) for res in results[0]]))

  # Alter timezone (IANA timezone)
  print(client.describe_collection(collection_name))
  print("===================alter collection timezone===================")
  try:
    client.alter_collection_properties(collection_name, {"timezone": "Asia/Shanghai"})
  except Exception as e:
    print(e)
  print(client.describe_collection(collection_name))

  try:
    client.alter_collection_properties(collection_name, {"timezone": "error"})
  except Exception as e:
    print(e)
  print(client.describe_collection(collection_name))

  try:
    client.alter_database_properties("default", {"timezone": "error"})
  except Exception as e:
    print(e)
  print(client.describe_database("default"))

  # Query with new operator
  results = client.query(
    collection_name, limit=10, timezone="Asia/Shanghai"
  )
  print("\n".join([str(res) for res in results]))

  expr = "tsz + INTERVAL 'P0D' != ISO '2025-01-03T00:00:00+08:00'"
  results = client.query(collection_name, expr, output_fields=["id", "tsz"], limit=10)
  print("  The first expr:")
  print("\n".join([str(res) for res in results]))
  expr = "tsz != ISO '2025-01-03T00:00:00+08:00'"
  print("  The second expr")
  results = client.query(collection_name, expr, output_fields=["id", "tsz"], limit=10)
  print("\n".join([str(res) for res in results]))

if __name__ == "__main__":
  main()
