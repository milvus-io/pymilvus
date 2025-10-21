import struct
from typing import List, Optional

from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2


def create_search_results(
    num_queries: int,
    top_k: int,
    output_fields: Optional[List[str]] = None,
    include_vectors: bool = False,
    dim: int = 128
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))
            elif field_name == "age":
                field_data.type = schema_pb2.DataType.Int32
                field_data.scalars.int_data.data.extend([25 + i % 50 for i in range(total_results)])
            elif field_name == "score":
                field_data.type = schema_pb2.DataType.Float
                field_data.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(total_results)])
            elif field_name == "name":
                field_data.type = schema_pb2.DataType.VarChar
                field_data.scalars.string_data.data.extend([f"name_{i}" for i in range(total_results)])
            elif field_name == "embedding" and include_vectors:
                field_data.type = schema_pb2.DataType.FloatVector
                field_data.vectors.dim = dim
                flat_vector = [float(j % 100) / 100.0 for _ in range(total_results) for j in range(dim)]
                field_data.vectors.float_vector.data.extend(flat_vector)

    return response


def create_search_results_with_float16_vector(
    num_queries: int,
    top_k: int,
    dim: int = 128,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "embedding":
                field_data.type = schema_pb2.DataType.Float16Vector
                field_data.vectors.dim = dim
                field_data.vectors.float16_vector = b'\x00' * (total_results * dim * 2)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_bfloat16_vector(
    num_queries: int,
    top_k: int,
    dim: int = 128,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "embedding":
                field_data.type = schema_pb2.DataType.BFloat16Vector
                field_data.vectors.dim = dim
                field_data.vectors.bfloat16_vector = b'\x00' * (total_results * dim * 2)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_binary_vector(
    num_queries: int,
    top_k: int,
    dim: int = 128,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "embedding":
                field_data.type = schema_pb2.DataType.BinaryVector
                field_data.vectors.dim = dim
                field_data.vectors.binary_vector = b'\x00' * (total_results * dim // 8)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_int8_vector(
    num_queries: int,
    top_k: int,
    dim: int = 128,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "embedding":
                field_data.type = schema_pb2.DataType.Int8Vector
                field_data.vectors.dim = dim
                field_data.vectors.int8_vector = b'\x00' * (total_results * dim)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_sparse_vector(
    num_queries: int,
    top_k: int
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

    field_data = results.fields_data.add()
    field_data.field_name = "sparse_embedding"
    field_data.type = schema_pb2.DataType.SparseFloatVector

    for _ in range(total_results):
        # Sparse format: index (uint32) + value (float32) pairs
        sparse_bytes = struct.pack('<I', 10) + struct.pack('<f', 0.5)
        field_data.vectors.sparse_float_vector.contents.append(sparse_bytes)
    field_data.vectors.sparse_float_vector.dim = 1000

    return response


def create_search_results_with_varchar(
    num_queries: int,
    top_k: int,
    varchar_length: int = 10,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "text":
                field_data.type = schema_pb2.DataType.VarChar
                field_data.scalars.string_data.data.extend(
                    ['x' * varchar_length] * total_results
                )
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_json(
    num_queries: int,
    top_k: int,
    json_size: str = "small",
    output_fields: Optional[List[str]] = None
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

    if json_size == "small":
        json_data = b'{"key": "value"}'
    elif json_size == "medium":
        json_data = b'{"name": "test", "age": 25, "tags": ["a", "b", "c"], "active": true}'
    elif json_size == "large":
        json_data = b'{"name": "test", "description": "' + b'x' * 500 + b'", "metadata": {"field1": 1, "field2": 2}}'
    else:  # huge ~64KB
        payload = b'x' * 65536
        json_data = b'{"blob": "' + payload + b'"}'

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "metadata":
                field_data.type = schema_pb2.DataType.JSON
                field_data.scalars.json_data.data.extend([json_data] * total_results)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_array(
    num_queries: int,
    top_k: int,
    array_len: int = 5,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "tags":
                field_data.type = schema_pb2.DataType.Array
                field_data.scalars.array_data.element_type = schema_pb2.DataType.Int64
                for _ in range(total_results):
                    array_item = field_data.scalars.array_data.data.add()
                    # Fill with zeros to avoid excessive memory overhead in test logic
                    array_item.long_data.data.extend([0] * array_len)
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_search_results_with_geojson(
    num_queries: int,
    top_k: int,
    output_fields: Optional[List[str]] = None
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

    if output_fields:
        results.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = results.fields_data.add()
            field_data.field_name = field_name

            if field_name == "location":
                field_data.type = schema_pb2.DataType.Geometry
                field_data.scalars.geometry_wkt_data.data.extend(
                    ["POINT(0.0 0.0)"] * total_results
                )
            elif field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(total_results)))

    return response


def create_query_results(
    num_rows: int,
    output_fields: Optional[List[str]] = None
) -> milvus_pb2.QueryResults:
    response = milvus_pb2.QueryResults()
    response.status.error_code = common_pb2.ErrorCode.Success

    if output_fields:
        response.output_fields.extend(output_fields)

        for field_name in output_fields:
            field_data = response.fields_data.add()
            field_data.field_name = field_name

            if field_name == "id":
                field_data.type = schema_pb2.DataType.Int64
                field_data.scalars.long_data.data.extend(list(range(num_rows)))
            elif field_name == "age":
                field_data.type = schema_pb2.DataType.Int32
                field_data.scalars.int_data.data.extend([25 + i % 50 for i in range(num_rows)])
            elif field_name == "score":
                field_data.type = schema_pb2.DataType.Float
                field_data.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(num_rows)])
            elif field_name == "name":
                field_data.type = schema_pb2.DataType.VarChar
                field_data.scalars.string_data.data.extend([f"name_{i}" for i in range(num_rows)])
            elif field_name == "active":
                field_data.type = schema_pb2.DataType.Bool
                field_data.scalars.bool_data.data.extend([i % 2 == 0 for i in range(num_rows)])
            elif field_name == "metadata":
                field_data.type = schema_pb2.DataType.JSON
                field_data.scalars.json_data.data.extend([b'{"key": "value"}'] * num_rows)

    return response


def create_hybrid_search_results(
    num_requests: int = 2,
    top_k: int = 10,
    output_fields: Optional[List[str]] = None
) -> milvus_pb2.SearchResults:
    return create_search_results(
        num_queries=1,
        top_k=top_k,
        output_fields=output_fields,
        include_vectors=False
    )


def create_search_results_all_types(
    num_queries: int = 1,
    top_k: int = 10,
    dim: int = 128
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

    output_fields = [
        "int8_field",
        "int16_field",
        "int32_field",
        "int64_field",
        "float_field",
        "double_field",
        "bool_field",
        "varchar_field",
        "json_field",
        "array_field",
        "geojson_field",
        "struct_field",
        "float_vector",
        "float16_vector",
        "bfloat16_vector",
        "binary_vector",
        "sparse_vector",
        "int8_vector",
    ]
    results.output_fields.extend(output_fields)

    # Int8 field
    field_data = results.fields_data.add()
    field_data.field_name = "int8_field"
    field_data.type = schema_pb2.DataType.Int8
    field_data.scalars.int_data.data.extend([i % 128 for i in range(total_results)])

    # Int16 field
    field_data = results.fields_data.add()
    field_data.field_name = "int16_field"
    field_data.type = schema_pb2.DataType.Int16
    field_data.scalars.int_data.data.extend([i % 1000 for i in range(total_results)])

    # Int32 field
    field_data = results.fields_data.add()
    field_data.field_name = "int32_field"
    field_data.type = schema_pb2.DataType.Int32
    field_data.scalars.int_data.data.extend(list(range(total_results)))

    # Int64 field
    field_data = results.fields_data.add()
    field_data.field_name = "int64_field"
    field_data.type = schema_pb2.DataType.Int64
    field_data.scalars.long_data.data.extend([i * 1000 for i in range(total_results)])

    # Float field
    field_data = results.fields_data.add()
    field_data.field_name = "float_field"
    field_data.type = schema_pb2.DataType.Float
    field_data.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(total_results)])

    # Double field
    field_data = results.fields_data.add()
    field_data.field_name = "double_field"
    field_data.type = schema_pb2.DataType.Double
    field_data.scalars.double_data.data.extend([0.123456789 + i for i in range(total_results)])

    # Bool field
    field_data = results.fields_data.add()
    field_data.field_name = "bool_field"
    field_data.type = schema_pb2.DataType.Bool
    field_data.scalars.bool_data.data.extend([i % 2 == 0 for i in range(total_results)])

    # VarChar field
    field_data = results.fields_data.add()
    field_data.field_name = "varchar_field"
    field_data.type = schema_pb2.DataType.VarChar
    field_data.scalars.string_data.data.extend([f"text_{i}" for i in range(total_results)])

    # JSON field
    field_data = results.fields_data.add()
    field_data.field_name = "json_field"
    field_data.type = schema_pb2.DataType.JSON
    field_data.scalars.json_data.data.extend([b'{"id": %d}' % i for i in range(total_results)])

    # Array field
    field_data = results.fields_data.add()
    field_data.field_name = "array_field"
    field_data.type = schema_pb2.DataType.Array
    field_data.scalars.array_data.element_type = schema_pb2.DataType.Int64
    for i in range(total_results):
        array_item = field_data.scalars.array_data.data.add()
        array_item.long_data.data.extend([i, i+1, i+2])

    # GeoJSON field
    field_data = results.fields_data.add()
    field_data.field_name = "geojson_field"
    field_data.type = schema_pb2.DataType.Geometry
    field_data.scalars.geometry_wkt_data.data.extend(
        [f"POINT({i}.0 {i}.0)" for i in range(total_results)]
    )

    # Float vector
    field_data = results.fields_data.add()
    field_data.field_name = "float_vector"
    field_data.type = schema_pb2.DataType.FloatVector
    field_data.vectors.dim = dim
    flat_vector = [float(j % 100) / 100.0 for _ in range(total_results) for j in range(dim)]
    field_data.vectors.float_vector.data.extend(flat_vector)

    # Float16 vector
    field_data = results.fields_data.add()
    field_data.field_name = "float16_vector"
    field_data.type = schema_pb2.DataType.Float16Vector
    field_data.vectors.dim = dim
    field_data.vectors.float16_vector = b'\x00' * (total_results * dim * 2)

    # BFloat16 vector
    field_data = results.fields_data.add()
    field_data.field_name = "bfloat16_vector"
    field_data.type = schema_pb2.DataType.BFloat16Vector
    field_data.vectors.dim = dim
    field_data.vectors.bfloat16_vector = b'\x00' * (total_results * dim * 2)

    # Binary vector
    field_data = results.fields_data.add()
    field_data.field_name = "binary_vector"
    field_data.type = schema_pb2.DataType.BinaryVector
    field_data.vectors.dim = dim
    field_data.vectors.binary_vector = b'\x00' * (total_results * dim // 8)

    # Sparse vector
    field_data = results.fields_data.add()
    field_data.field_name = "sparse_vector"
    field_data.type = schema_pb2.DataType.SparseFloatVector
    for _ in range(total_results):
        # Sparse format: index (uint32) + value (float32) pairs
        # Create one sparse entry: index 10 with value 0.5
        sparse_bytes = struct.pack('<I', 10) + struct.pack('<f', 0.5)
        field_data.vectors.sparse_float_vector.contents.append(sparse_bytes)
    field_data.vectors.sparse_float_vector.dim = 1000

    # Int8 vector
    field_data = results.fields_data.add()
    field_data.field_name = "int8_vector"
    field_data.type = schema_pb2.DataType.Int8Vector
    field_data.vectors.dim = dim
    field_data.vectors.int8_vector = b'\x00' * (total_results * dim)

    # Struct field (stored as ARRAY_OF_STRUCT internally)
    field_data = results.fields_data.add()
    field_data.field_name = "struct_field"
    field_data.type = schema_pb2.ArrayOfStruct

    # Create sub-field for int data (ARRAY type)
    sub_field_int = field_data.struct_arrays.fields.add()
    sub_field_int.field_name = "sub_int"
    sub_field_int.type = schema_pb2.Array
    sub_field_int.scalars.array_data.element_type = schema_pb2.Int64
    for i in range(total_results):
        array_item = sub_field_int.scalars.array_data.data.add()
        array_item.long_data.data.extend([i * 10, i * 10 + 1])

    # Create sub-field for string data (ARRAY type)
    sub_field_str = field_data.struct_arrays.fields.add()
    sub_field_str.field_name = "sub_str"
    sub_field_str.type = schema_pb2.Array
    sub_field_str.scalars.array_data.element_type = schema_pb2.VarChar
    for i in range(total_results):
        array_item = sub_field_str.scalars.array_data.data.add()
        array_item.string_data.data.extend([f"struct_{i}_0", f"struct_{i}_1"])

    return response


def create_query_results_all_types(
    num_rows: int,
    dim: int = 128
) -> milvus_pb2.QueryResults:
    response = milvus_pb2.QueryResults()
    response.status.error_code = common_pb2.ErrorCode.Success

    output_fields = [
        "int8_field", "int16_field", "int32_field", "int64_field",
        "float_field", "double_field", "bool_field", "varchar_field",
        "json_field", "array_field", "geojson_field", "struct_field"
    ]
    response.output_fields.extend(output_fields)

    # Copy all scalar field logic from search version
    for field_name in output_fields:
        field_data = response.fields_data.add()
        field_data.field_name = field_name

        if field_name == "int8_field":
            field_data.type = schema_pb2.DataType.Int8
            field_data.scalars.int_data.data.extend([i % 128 for i in range(num_rows)])
        elif field_name == "int16_field":
            field_data.type = schema_pb2.DataType.Int16
            field_data.scalars.int_data.data.extend([i % 1000 for i in range(num_rows)])
        elif field_name == "int32_field":
            field_data.type = schema_pb2.DataType.Int32
            field_data.scalars.int_data.data.extend(list(range(num_rows)))
        elif field_name == "int64_field":
            field_data.type = schema_pb2.DataType.Int64
            field_data.scalars.long_data.data.extend([i * 1000 for i in range(num_rows)])
        elif field_name == "float_field":
            field_data.type = schema_pb2.DataType.Float
            field_data.scalars.float_data.data.extend([0.5 + i * 0.01 for i in range(num_rows)])
        elif field_name == "double_field":
            field_data.type = schema_pb2.DataType.Double
            field_data.scalars.double_data.data.extend([0.123456789 + i for i in range(num_rows)])
        elif field_name == "bool_field":
            field_data.type = schema_pb2.DataType.Bool
            field_data.scalars.bool_data.data.extend([i % 2 == 0 for i in range(num_rows)])
        elif field_name == "varchar_field":
            field_data.type = schema_pb2.DataType.VarChar
            field_data.scalars.string_data.data.extend([f"text_{i}" for i in range(num_rows)])
        elif field_name == "json_field":
            field_data.type = schema_pb2.DataType.JSON
            field_data.scalars.json_data.data.extend([b'{"id": %d}' % i for i in range(num_rows)])
        elif field_name == "array_field":
            field_data.type = schema_pb2.DataType.Array
            field_data.scalars.array_data.element_type = schema_pb2.DataType.Int64
            for i in range(num_rows):
                array_item = field_data.scalars.array_data.data.add()
                array_item.long_data.data.extend([i, i+1, i+2])
        elif field_name == "geojson_field":
            field_data.type = schema_pb2.DataType.Geometry
            field_data.scalars.geometry_wkt_data.data.extend(
                [f"POINT({i}.0 {i}.0)" for i in range(num_rows)]
            )
        elif field_name == "struct_field":
            field_data.type = schema_pb2.ArrayOfStruct

            # Create sub-field for int data (ARRAY type)
            sub_field_int = field_data.struct_arrays.fields.add()
            sub_field_int.field_name = "sub_int"
            sub_field_int.type = schema_pb2.Array
            sub_field_int.scalars.array_data.element_type = schema_pb2.Int64
            for i in range(num_rows):
                array_item = sub_field_int.scalars.array_data.data.add()
                array_item.long_data.data.extend([i * 10, i * 10 + 1])

            # Create sub-field for string data (ARRAY type)
            sub_field_str = field_data.struct_arrays.fields.add()
            sub_field_str.field_name = "sub_str"
            sub_field_str.type = schema_pb2.Array
            sub_field_str.scalars.array_data.element_type = schema_pb2.VarChar
            for i in range(num_rows):
                array_item = sub_field_str.scalars.array_data.data.add()
                array_item.string_data.data.extend([f"struct_{i}_0", f"struct_{i}_1"])

    return response
