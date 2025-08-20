from pymilvus import MilvusClient, DataType, CollectionSchema, IndexType, FieldSchema, milvus_client
import datetime
import pytz

milvus_host = "http://localhost:19530"
collection_name = "timestamptz_test123"
uri = "https://in03-310f8892c9ce201.serverless.aws-eu-central-1.cloud.zilliz.com"
token = "5fe592ff63a89cd951638a84e6314f421556e8a74ff4776e9761468e7a5e0a3e27750b493c079c7c56304f6cd8c5609323af7497"


def main():
  client = MilvusClient(uri=milvus_host)
  # client = MilvusClient(uri=uri, token=token)

  # Create collection with TIMESTAMPTZ field
  if client.has_collection(collection_name):
    client.drop_collection(collection_name)

  schema = client.create_schema()
  schema.add_field("id", DataType.INT64, is_primary=True)
  schema.add_field("tsz", DataType.TIMESTAMPTZ)
  schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
  client.create_collection(collection_name, schema=schema, consistency_level="Session")
  index_param = client.prepare_index_params(
    collection_name=collection_name,
    field_name="vec",
    index_type=IndexType.HNSW,
    metric_type="COSINE",
    params={"M": 30, "efConstruction": 200},
  )
  client.create_index(collection_name, index_param)
  print(client.describe_collection(collection_name))
  client.load_collection(collection_name)
  print(f"load state: {client.get_load_state(collection_name)}")

  # Insert data with timezone-aware timestamps
  print("===================insert timestamptz===================")
  data_size = 10
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
  )
  print("\n".join([str(res) for res in results]))

  print("====================test search====================")
  results = client.search(
    collection_name,
    [[0.5, 0.6, 0.7, 0.8]],
    output_fields=["id", "tsz", "vec"],
    limit=2,
    timezone="America/Chicago",
  )
  print("\n".join([str(res) for res in results[0]]))

  # Alter timezone (IANA timezone)
  print("===================alter collection timezone===================")
  try:
    client.alter_collection_properties(collection_name, {"collection.timezone": "Asia/Shanghai"})
  except Exception as e:
    print(e)
  print(client.describe_collection(collection_name))

  try:
    client.alter_collection_properties(collection_name, {"collection.timezone": "error"})
  except Exception as e:
    print(e)
  print(client.describe_collection(collection_name))

  print("===================alter database timezone===================")
  try:
    client.alter_database_properties("default", {"database.timezone": "Asia/Shanghai"})
  except Exception as e:
    print(e)
  print(client.describe_database("default"))

  try:
    client.alter_database_properties("default", {"database.timezone": "error"})
  except Exception as e:
    print(e)
  print(client.describe_database("default"))

  # # Query with extract (e.g., extract hour from tsz)
  expr = "EXTRACT hour from tsz"
  results = client.query(collection_name, expr, output_fields=["id", "tsz"])
  print(results)

  # # Scalar filtering with comparison and interval
  # expr = "tsz + interval 'P1D' < to_timestamptz('2025-05-03 00:00:00+09:00')"
  # results = client.query(collection_name, expr, output_fields=["id", "tsz"])
  # print(results)

  # # Create index on TIMESTAMPTZ
  # index_params = client.prepare_index_params()
  # index_params.add_index(
  #   field_name="tsz",
  #   index_type="INVERTED",
  #   index_name="tsz_index",
  # )
  # client.create_index(collection_name, index_params)
  # # TODO: Add benchmark for query with TIMESTAMPTZ


if __name__ == "__main__":
  main()
