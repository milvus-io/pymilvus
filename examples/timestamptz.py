from pymilvus import MilvusClient, DataType, CollectionSchema, IndexType, FieldSchema, milvus_client
import datetime

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
  # schema.add_field("tsz", DataType.TIMESTAMPTZ)
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
  data_size = 10000
  data = [
    {
      "id": i + 1,
      "tsz": int((datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(days=i)).timestamp() * 1000),
      "vec": [float(i) / 10 for i in range(4)],
    }
    for i in range(data_size)
  ]
  insert_res = client.insert(collection_name, data)
  print(f"Insert result: {insert_res}")
  # Query data with TIMESTAMPTZ
  results = client.query(
    collection_name,
    filter="id == 1",
    output_fields=["id", "tsz", "vec"],
    limit=1,
  )
  print(results)

  # # Alter collection timezone (IANA timezone)
  # client.alter_collection_properties(collection_name, {"timezone": "Asia/Shanghai"})
  # client.alter_database_properties("default", {"timezone": "Asia/Shanghai"})

  # # Query with extract (e.g., extract hour from tsz)
  # expr = "extract(tsz, hour)"
  # results = client.query(collection_name, expr, output_fields=["id", "tsz"])
  # print(results)

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
