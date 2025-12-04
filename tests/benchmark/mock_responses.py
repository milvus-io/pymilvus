import struct
from typing import List, Optional

from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from pymilvus.orm.types import DataType


def create_search_results_from_schema(
    schema: CollectionSchema,
    num_queries: int,
    top_k: int,
    output_fields: Optional[List[str]] = None,
) -> milvus_pb2.SearchResults:
    response = milvus_pb2.SearchResults()
    response.status.error_code = common_pb2.ErrorCode.Success

    results = response.results
    results.num_queries = num_queries
    results.top_k = top_k

    total_results = num_queries * top_k
    results.ids.int_id.data.extend(list(range(total_results)))
    results.scores.extend([0.9 - i * 0.01 for i in range(total_results)])
    results.topks.extend([top_k] * num_queries)

    # Determine which fields to include
    if output_fields is None or len(output_fields) == 0 or output_fields == ["*"]:
        # Include all fields
        fields_to_include = schema.fields
    else:
        # Filter fields based on output_fields
        field_map = {f.name: f for f in schema.fields}
        fields_to_include = [field_map[name] for name in output_fields if name in field_map]

    # Generate field data based on CollectionSchema
    for field in fields_to_include:
        fd = results.fields_data.add()
        fd.field_name = field.name
        _fill_field_data(field, fd, total_results)
        results.output_fields.append(field.name)

    return response


def _fill_field_data(field: FieldSchema, dest, total_results: int) -> None:
    name = field.name
    dtype = field.dtype
    params = field.params or {}
    dim = params.get('dim', 128)
    max_length = params.get('max_length', 100)

    # Scalars
    if dtype == DataType.INT8:
        dest.type = schema_pb2.DataType.Int8
        dest.scalars.int_data.data.extend([i % 128 for i in range(total_results)])
    elif dtype == DataType.INT16:
        dest.type = schema_pb2.DataType.Int16
        dest.scalars.int_data.data.extend([i % 1000 for i in range(total_results)])
    elif dtype == DataType.INT32:
        dest.type = schema_pb2.DataType.Int32
        dest.scalars.int_data.data.extend(list(range(total_results)))
    elif dtype == DataType.INT64:
        dest.type = schema_pb2.DataType.Int64
        dest.scalars.long_data.data.extend(list(range(total_results)))
    elif dtype == DataType.FLOAT:
        dest.type = schema_pb2.DataType.Float
        dest.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(total_results)])
    elif dtype == DataType.DOUBLE:
        dest.type = schema_pb2.DataType.Double
        dest.scalars.double_data.data.extend([float(i) for i in range(total_results)])
    elif dtype == DataType.BOOL:
        dest.type = schema_pb2.DataType.Bool
        dest.scalars.bool_data.data.extend([i % 2 == 0 for i in range(total_results)])
    elif dtype == DataType.VARCHAR:
        dest.type = schema_pb2.DataType.VarChar
        data = []
        for i in range(total_results):
            base = f"{name}_{i}_"
            padding = 'x' * max(0, max_length - len(base))
            s = (base + padding)[:max_length]
            data.append(s)
        dest.scalars.string_data.data.extend(data)
    elif dtype == DataType.TIMESTAMPTZ:
        dest.type = schema_pb2.DataType.Timestamptz
        dest.scalars.string_data.data.extend([f"2024-01-01T00:00:{i:02d}Z" for i in range(total_results)])
    elif dtype == DataType.JSON:
        dest.type = schema_pb2.DataType.JSON
        data = []
        for i in range(total_results):
            base = b'{"i":%d,"d":"' % i
            remaining = max(0, max_length - len(base) - 2)  # -2 for closing "}
            padding = b'x' * remaining
            json_bytes = (base + padding + b'"}')[:max_length]
            data.append(json_bytes)
        dest.scalars.json_data.data.extend(data)
    elif dtype == DataType.GEOMETRY:
        dest.type = schema_pb2.DataType.Geometry
        dest.scalars.geometry_wkt_data.data.extend(["POINT(0 0)"] * total_results)
    elif dtype == DataType.ARRAY:
        dest.type = schema_pb2.DataType.Array
        dest.scalars.array_data.element_type = schema_pb2.DataType.Int64
        for i in range(total_results):
            item = dest.scalars.array_data.data.add()
            item.long_data.data.extend([i, i + 1, i + 2])
    # Vectors
    elif dtype == DataType.FLOAT_VECTOR:
        dest.type = schema_pb2.DataType.FloatVector
        dest.vectors.dim = dim
        flat = [float(j % 100) / 100.0 for _ in range(total_results) for j in range(dim)]
        dest.vectors.float_vector.data.extend(flat)
    elif dtype == DataType.FLOAT16_VECTOR:
        dest.type = schema_pb2.DataType.Float16Vector
        dest.vectors.dim = dim
        dest.vectors.float16_vector = b"\x00" * (total_results * dim * 2)
    elif dtype == DataType.BFLOAT16_VECTOR:
        dest.type = schema_pb2.DataType.BFloat16Vector
        dest.vectors.dim = dim
        dest.vectors.bfloat16_vector = b"\x00" * (total_results * dim * 2)
    elif dtype == DataType.BINARY_VECTOR:
        dest.type = schema_pb2.DataType.BinaryVector
        dest.vectors.dim = dim
        dest.vectors.binary_vector = b"\x00" * (total_results * dim // 8)
    elif dtype == DataType.INT8_VECTOR:
        dest.type = schema_pb2.DataType.Int8Vector
        dest.vectors.dim = dim
        dest.vectors.int8_vector = b"\x00" * (total_results * dim)
    elif dtype == DataType.SPARSE_FLOAT_VECTOR:
        dest.type = schema_pb2.DataType.SparseFloatVector
        for _ in range(total_results):
            sparse_bytes = struct.pack('<I', 10) + struct.pack('<f', 0.5)
            dest.vectors.sparse_float_vector.contents.append(sparse_bytes)
        dest.vectors.sparse_float_vector.dim = 1000
