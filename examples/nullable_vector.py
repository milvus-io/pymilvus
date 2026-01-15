import numpy as np
import ml_dtypes

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    MilvusClient,
)

DIM = 4

def gen_float_vector(dim):
    return np.random.rand(dim).astype(np.float32)


def gen_binary_vector(dim):
    return np.packbits(np.random.randint(0, 2, dim), axis=-1).tobytes()


def gen_fp16_vector(dim):
    return np.random.rand(dim).astype(np.float16)


def gen_bf16_vector(dim):
    return np.random.rand(dim).astype(ml_dtypes.bfloat16)


def gen_sparse_vector():
    dim = np.random.randint(2, 20)
    return {i: float(np.random.rand()) for i in range(dim)}


def gen_int8_vector(dim):
    return np.random.randint(-128, 127, dim, dtype=np.int8)


def vectors_equal(v1, v2, vtype_name):
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False

    if vtype_name == "sparse_float_vector":
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(abs(v1[k] - v2[k]) < 1e-5 for k in v1.keys())
    elif vtype_name == "binary_vector":
        b1 = v1[0] if isinstance(v1, list) else v1
        b2 = v2[0] if isinstance(v2, list) else v2
        return b1 == b2
    elif vtype_name in ("float16_vector", "bfloat16_vector"):
        dtype = np.float16 if vtype_name == "float16_vector" else ml_dtypes.bfloat16
        def to_array(v):
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            if isinstance(v, bytes):
                return np.frombuffer(v, dtype=dtype)
            return np.asarray(v, dtype=dtype)
        return np.allclose(to_array(v1).astype(np.float32), to_array(v2).astype(np.float32), rtol=1e-2)
    elif vtype_name == "int8_vector":
        def to_array(v):
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            if isinstance(v, bytes):
                return np.frombuffer(v, dtype=np.int8)
            return np.asarray(v, dtype=np.int8)
        return np.array_equal(to_array(v1), to_array(v2))
    else:
        return np.allclose(np.asarray(v1), np.asarray(v2), rtol=1e-3)


connections.connect("default", host="localhost", port="19530")

VECTOR_TYPES = [
    {
        "name": "float_vector",
        "dtype": DataType.FLOAT_VECTOR,
        "dim": DIM,
        "gen_vector": lambda: gen_float_vector(DIM),
        "index_params": {"metric_type": "L2", "index_type": "FLAT", "params": {}},
    },
    {
        "name": "binary_vector",
        "dtype": DataType.BINARY_VECTOR,
        "dim": DIM * 8,
        "gen_vector": lambda: gen_binary_vector(DIM * 8),
        "index_params": {"metric_type": "HAMMING", "index_type": "BIN_FLAT", "params": {}},
    },
    {
        "name": "float16_vector",
        "dtype": DataType.FLOAT16_VECTOR,
        "dim": DIM,
        "gen_vector": lambda: gen_fp16_vector(DIM),
        "index_params": {"metric_type": "L2", "index_type": "FLAT", "params": {}},
    },
    {
        "name": "bfloat16_vector",
        "dtype": DataType.BFLOAT16_VECTOR,
        "dim": DIM,
        "gen_vector": lambda: gen_bf16_vector(DIM),
        "index_params": {"metric_type": "L2", "index_type": "FLAT", "params": {}},
    },
    {
        "name": "sparse_float_vector",
        "dtype": DataType.SPARSE_FLOAT_VECTOR,
        "dim": None,
        "gen_vector": gen_sparse_vector,
        "index_params": {"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}},
    },
    {
        "name": "int8_vector",
        "dtype": DataType.INT8_VECTOR,
        "dim": DIM,
        "gen_vector": lambda: gen_int8_vector(DIM),
        "index_params": {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}},
    },
]


def test_nullable_vector(vector_config, batch_size=1000, num_batches=10, null_percent=10):
    vtype_name = vector_config["name"]
    dtype = vector_config["dtype"]
    dim = vector_config["dim"]
    gen_vector = vector_config["gen_vector"]
    index_params = vector_config["index_params"]
    collection_name = f"test_nullable_{vtype_name}"

    print(f"\n[Test] {vtype_name} ({batch_size}x{num_batches} rows, {null_percent}% null)")

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    ]
    if dim is not None:
        fields.append(FieldSchema(name="embedding", dtype=dtype, dim=dim, nullable=True))
    else:
        fields.append(FieldSchema(name="embedding", dtype=dtype, nullable=True))

    collection = Collection(collection_name, CollectionSchema(fields))
    collection.create_index("embedding", index_params)

    expected = {}
    current_id = 0

    for batch_idx in range(num_batches):
        data = []
        for i in range(batch_size):
            row_id = current_id + i
            is_null = (row_id % 100) < null_percent
            embedding = None if is_null else gen_vector()
            expected[row_id] = embedding
            data.append({"id": row_id, "name": f"row_{row_id}", "embedding": embedding})

        current_id += batch_size
        collection.insert(data)
        collection.flush()
        collection.load()

        results = collection.query(expr="id >= 0", output_fields=["id", "embedding"], limit=current_id + 1000)
        result_map = {row['id']: row.get('embedding') for row in results}

        assert len(results) == current_id
        for row_id, exp in expected.items():
            actual = result_map[row_id]
            if exp is None:
                assert actual is None, f"id={row_id}: expected None"
            else:
                assert actual is not None and vectors_equal(actual, exp, vtype_name), f"id={row_id}: mismatch"

        search_results = collection.search(
            data=[gen_vector()], anns_field="embedding",
            param={"metric_type": index_params["metric_type"]},
            limit=min(100, current_id), output_fields=["id", "embedding"]
        )
        for hit in search_results[0]:
            assert hit.entity.get('embedding') is not None
            assert expected[hit.id] is not None

        collection.release()
        null_count = sum(1 for v in expected.values() if v is None)
        print(f"  Batch {batch_idx+1}/{num_batches}: {len(expected)-null_count} valid, {null_count} null - OK")

    utility.drop_collection(collection_name)
    null_count = sum(1 for v in expected.values() if v is None)
    print(f"  PASSED ({len(expected)-null_count} valid, {null_count} null)")
    return True


def test_add_nullable_vector_column(vector_config, initial_rows=10, new_rows=10, null_percent=50):
    vtype_name = vector_config["name"]
    dtype = vector_config["dtype"]
    dim = vector_config["dim"]
    gen_vector = vector_config["gen_vector"]
    index_params = vector_config["index_params"]
    collection_name = f"test_add_col_{vtype_name}"

    print(f"\n[Test Add Column] {vtype_name} ({initial_rows} initial + {new_rows} new, {null_percent}% null)")

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="base_vec", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    collection = Collection(collection_name, CollectionSchema(fields))

    initial_data = [
        {"id": i, "name": f"initial_{i}", "base_vec": [float(i), float(i+1), float(i+2), float(i+3)]}
        for i in range(initial_rows)
    ]
    collection.insert(initial_data)
    collection.flush()

    client = MilvusClient(uri="http://localhost:19530")
    if dim is not None:
        client.add_collection_field(collection_name, "embedding", dtype, dim=dim, nullable=True)
    else:
        client.add_collection_field(collection_name, "embedding", dtype, nullable=True)
    collection = Collection(collection_name)

    collection.create_index("base_vec", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    collection.create_index("embedding", index_params)

    expected = {i: None for i in range(initial_rows)}

    data = []
    for i in range(new_rows):
        row_id = initial_rows + i
        is_null = ((i * 100) // new_rows) < null_percent
        embedding = None if is_null else gen_vector()
        expected[row_id] = embedding
        data.append({
            "id": row_id, "name": f"new_{row_id}",
            "base_vec": [float(row_id), float(row_id+1), float(row_id+2), float(row_id+3)],
            "embedding": embedding
        })

    collection.insert(data)
    collection.flush()
    collection.load()

    results = collection.query(expr="id >= 0", output_fields=["id", "embedding"], limit=initial_rows + new_rows + 100)
    result_map = {row['id']: row.get('embedding') for row in results}

    assert len(results) == initial_rows + new_rows
    for row_id, exp in expected.items():
        actual = result_map[row_id]
        if exp is None:
            assert actual is None, f"id={row_id}: expected None"
        else:
            assert actual is not None and vectors_equal(actual, exp, vtype_name), f"id={row_id}: mismatch"

    valid_count = sum(1 for v in expected.values() if v is not None)
    if valid_count > 0:
        search_results = collection.search(
            data=[gen_vector()], anns_field="embedding",
            param={"metric_type": index_params["metric_type"]},
            limit=min(100, valid_count), output_fields=["id", "embedding"]
        )
        for hit in search_results[0]:
            assert hit.entity.get('embedding') is not None
            assert expected[hit.id] is not None
            assert hit.id >= initial_rows

    collection.release()
    utility.drop_collection(collection_name)
    null_count = sum(1 for v in expected.values() if v is None)
    print(f"  PASSED ({valid_count} valid, {null_count} null)")
    return True


def main():
    print("Nullable Vector Test")

    passed, failed = [], []
    for config in VECTOR_TYPES:
        try:
            if test_nullable_vector(config):
                passed.append(config["name"])
            else:
                failed.append(config["name"])
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(config["name"])

    for config in VECTOR_TYPES:
        try:
            if test_add_nullable_vector_column(config):
                passed.append(f"add_col_{config['name']}")
            else:
                failed.append(f"add_col_{config['name']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(f"add_col_{config['name']}")

    print(f"\nSummary: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    connections.disconnect("default")
    return len(failed) == 0


if __name__ == "__main__":
    exit(0 if main() else 1)
