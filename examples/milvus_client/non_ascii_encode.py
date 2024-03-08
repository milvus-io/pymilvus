import numpy as np
from pymilvus import MilvusClient, DataType

dimension = 128
collection_name = "books"
client = MilvusClient("http://localhost:19530")
client.drop_collection(collection_name)

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dimension)
schema.add_field("info", DataType.JSON)

index_params = client.prepare_index_params("embeddings", metric_type="L2")
client.create_collection(collection_name, schema=schema, index_params=index_params)

rng = np.random.default_rng(seed=19530)
rows = [
    {"embeddings": rng.random((1, dimension))[0],
     "info": {"title": "Lord of the Flies", "author": "William Golding"}},

    {"embeddings": rng.random((1, dimension))[0],
     "info": {"作者": "J.D.塞林格", "title": "麦田里的守望者", }},

    {"embeddings": rng.random((1, dimension))[0],
     "info": {"Título": "Cien años de soledad", "autor": "Gabriel García Márquez"}},
]

client.insert(collection_name, rows)
result = client.query(collection_name, filter="info['作者'] == 'J.D.塞林格' or info['Título'] == 'Cien años de soledad'",
                      output_fields=["info"],
                      consistency_level="Strong")

for hit in result:
    print(f"hit: {hit}")
