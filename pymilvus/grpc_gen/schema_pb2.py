# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: schema.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2
from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cschema.proto\x12\x13milvus.proto.schema\x1a\x0c\x63ommon.proto\x1a google/protobuf/descriptor.proto\"\xd7\x03\n\x0b\x46ieldSchema\x12\x0f\n\x07\x66ieldID\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x16\n\x0eis_primary_key\x18\x03 \x01(\x08\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x30\n\tdata_type\x18\x05 \x01(\x0e\x32\x1d.milvus.proto.schema.DataType\x12\x36\n\x0btype_params\x18\x06 \x03(\x0b\x32!.milvus.proto.common.KeyValuePair\x12\x37\n\x0cindex_params\x18\x07 \x03(\x0b\x32!.milvus.proto.common.KeyValuePair\x12\x0e\n\x06\x61utoID\x18\x08 \x01(\x08\x12.\n\x05state\x18\t \x01(\x0e\x32\x1f.milvus.proto.schema.FieldState\x12\x33\n\x0c\x65lement_type\x18\n \x01(\x0e\x32\x1d.milvus.proto.schema.DataType\x12\x36\n\rdefault_value\x18\x0b \x01(\x0b\x32\x1f.milvus.proto.schema.ValueField\x12\x12\n\nis_dynamic\x18\x0c \x01(\x08\x12\x18\n\x10is_partition_key\x18\r \x01(\x08\"\x99\x01\n\x10\x43ollectionSchema\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x12\n\x06\x61utoID\x18\x03 \x01(\x08\x42\x02\x18\x01\x12\x30\n\x06\x66ields\x18\x04 \x03(\x0b\x32 .milvus.proto.schema.FieldSchema\x12\x1c\n\x14\x65nable_dynamic_field\x18\x05 \x01(\x08\"\x19\n\tBoolArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x08\"\x18\n\x08IntArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\"\x19\n\tLongArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x03\"\x1a\n\nFloatArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\"\x1b\n\x0b\x44oubleArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x01\"\x1a\n\nBytesArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x0c\"\x1b\n\x0bStringArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\t\"q\n\nArrayArray\x12.\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32 .milvus.proto.schema.ScalarField\x12\x33\n\x0c\x65lement_type\x18\x02 \x01(\x0e\x32\x1d.milvus.proto.schema.DataType\"\x19\n\tJSONArray\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x0c\"\xac\x01\n\nValueField\x12\x13\n\tbool_data\x18\x01 \x01(\x08H\x00\x12\x12\n\x08int_data\x18\x02 \x01(\x05H\x00\x12\x13\n\tlong_data\x18\x03 \x01(\x03H\x00\x12\x14\n\nfloat_data\x18\x04 \x01(\x02H\x00\x12\x15\n\x0b\x64ouble_data\x18\x05 \x01(\x01H\x00\x12\x15\n\x0bstring_data\x18\x06 \x01(\tH\x00\x12\x14\n\nbytes_data\x18\x07 \x01(\x0cH\x00\x42\x06\n\x04\x64\x61ta\"\xfe\x03\n\x0bScalarField\x12\x33\n\tbool_data\x18\x01 \x01(\x0b\x32\x1e.milvus.proto.schema.BoolArrayH\x00\x12\x31\n\x08int_data\x18\x02 \x01(\x0b\x32\x1d.milvus.proto.schema.IntArrayH\x00\x12\x33\n\tlong_data\x18\x03 \x01(\x0b\x32\x1e.milvus.proto.schema.LongArrayH\x00\x12\x35\n\nfloat_data\x18\x04 \x01(\x0b\x32\x1f.milvus.proto.schema.FloatArrayH\x00\x12\x37\n\x0b\x64ouble_data\x18\x05 \x01(\x0b\x32 .milvus.proto.schema.DoubleArrayH\x00\x12\x37\n\x0bstring_data\x18\x06 \x01(\x0b\x32 .milvus.proto.schema.StringArrayH\x00\x12\x35\n\nbytes_data\x18\x07 \x01(\x0b\x32\x1f.milvus.proto.schema.BytesArrayH\x00\x12\x35\n\narray_data\x18\x08 \x01(\x0b\x32\x1f.milvus.proto.schema.ArrayArrayH\x00\x12\x33\n\tjson_data\x18\t \x01(\x0b\x32\x1e.milvus.proto.schema.JSONArrayH\x00\x42\x06\n\x04\x64\x61ta\"\xa9\x01\n\x0bVectorField\x12\x0b\n\x03\x64im\x18\x01 \x01(\x03\x12\x37\n\x0c\x66loat_vector\x18\x02 \x01(\x0b\x32\x1f.milvus.proto.schema.FloatArrayH\x00\x12\x17\n\rbinary_vector\x18\x03 \x01(\x0cH\x00\x12\x18\n\x0e\x66loat16_vector\x18\x04 \x01(\x0cH\x00\x12\x19\n\x0f\x62\x66loat16_vector\x18\x05 \x01(\x0cH\x00\x42\x06\n\x04\x64\x61ta\"\xe5\x01\n\tFieldData\x12+\n\x04type\x18\x01 \x01(\x0e\x32\x1d.milvus.proto.schema.DataType\x12\x12\n\nfield_name\x18\x02 \x01(\t\x12\x33\n\x07scalars\x18\x03 \x01(\x0b\x32 .milvus.proto.schema.ScalarFieldH\x00\x12\x33\n\x07vectors\x18\x04 \x01(\x0b\x32 .milvus.proto.schema.VectorFieldH\x00\x12\x10\n\x08\x66ield_id\x18\x05 \x01(\x03\x12\x12\n\nis_dynamic\x18\x06 \x01(\x08\x42\x07\n\x05\x66ield\"w\n\x03IDs\x12\x30\n\x06int_id\x18\x01 \x01(\x0b\x32\x1e.milvus.proto.schema.LongArrayH\x00\x12\x32\n\x06str_id\x18\x02 \x01(\x0b\x32 .milvus.proto.schema.StringArrayH\x00\x42\n\n\x08id_field\"\x86\x02\n\x10SearchResultData\x12\x13\n\x0bnum_queries\x18\x01 \x01(\x03\x12\r\n\x05top_k\x18\x02 \x01(\x03\x12\x33\n\x0b\x66ields_data\x18\x03 \x03(\x0b\x32\x1e.milvus.proto.schema.FieldData\x12\x0e\n\x06scores\x18\x04 \x03(\x02\x12%\n\x03ids\x18\x05 \x01(\x0b\x32\x18.milvus.proto.schema.IDs\x12\r\n\x05topks\x18\x06 \x03(\x03\x12\x15\n\routput_fields\x18\x07 \x03(\t\x12<\n\x14group_by_field_value\x18\x08 \x01(\x0b\x32\x1e.milvus.proto.schema.FieldData*\xd8\x01\n\x08\x44\x61taType\x12\x08\n\x04None\x10\x00\x12\x08\n\x04\x42ool\x10\x01\x12\x08\n\x04Int8\x10\x02\x12\t\n\x05Int16\x10\x03\x12\t\n\x05Int32\x10\x04\x12\t\n\x05Int64\x10\x05\x12\t\n\x05\x46loat\x10\n\x12\n\n\x06\x44ouble\x10\x0b\x12\n\n\x06String\x10\x14\x12\x0b\n\x07VarChar\x10\x15\x12\t\n\x05\x41rray\x10\x16\x12\x08\n\x04JSON\x10\x17\x12\x10\n\x0c\x42inaryVector\x10\x64\x12\x0f\n\x0b\x46loatVector\x10\x65\x12\x11\n\rFloat16Vector\x10\x66\x12\x12\n\x0e\x42\x46loat16Vector\x10g*V\n\nFieldState\x12\x10\n\x0c\x46ieldCreated\x10\x00\x12\x11\n\rFieldCreating\x10\x01\x12\x11\n\rFieldDropping\x10\x02\x12\x10\n\x0c\x46ieldDropped\x10\x03\x42m\n\x0eio.milvus.grpcB\x0bSchemaProtoP\x01Z4github.com/milvus-io/milvus-proto/go-api/v2/schemapb\xa0\x01\x01\xaa\x02\x12Milvus.Client.Grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'schema_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\016io.milvus.grpcB\013SchemaProtoP\001Z4github.com/milvus-io/milvus-proto/go-api/v2/schemapb\240\001\001\252\002\022Milvus.Client.Grpc'
  _globals['_COLLECTIONSCHEMA'].fields_by_name['autoID']._options = None
  _globals['_COLLECTIONSCHEMA'].fields_by_name['autoID']._serialized_options = b'\030\001'
  _globals['_DATATYPE']._serialized_start=2530
  _globals['_DATATYPE']._serialized_end=2746
  _globals['_FIELDSTATE']._serialized_start=2748
  _globals['_FIELDSTATE']._serialized_end=2834
  _globals['_FIELDSCHEMA']._serialized_start=86
  _globals['_FIELDSCHEMA']._serialized_end=557
  _globals['_COLLECTIONSCHEMA']._serialized_start=560
  _globals['_COLLECTIONSCHEMA']._serialized_end=713
  _globals['_BOOLARRAY']._serialized_start=715
  _globals['_BOOLARRAY']._serialized_end=740
  _globals['_INTARRAY']._serialized_start=742
  _globals['_INTARRAY']._serialized_end=766
  _globals['_LONGARRAY']._serialized_start=768
  _globals['_LONGARRAY']._serialized_end=793
  _globals['_FLOATARRAY']._serialized_start=795
  _globals['_FLOATARRAY']._serialized_end=821
  _globals['_DOUBLEARRAY']._serialized_start=823
  _globals['_DOUBLEARRAY']._serialized_end=850
  _globals['_BYTESARRAY']._serialized_start=852
  _globals['_BYTESARRAY']._serialized_end=878
  _globals['_STRINGARRAY']._serialized_start=880
  _globals['_STRINGARRAY']._serialized_end=907
  _globals['_ARRAYARRAY']._serialized_start=909
  _globals['_ARRAYARRAY']._serialized_end=1022
  _globals['_JSONARRAY']._serialized_start=1024
  _globals['_JSONARRAY']._serialized_end=1049
  _globals['_VALUEFIELD']._serialized_start=1052
  _globals['_VALUEFIELD']._serialized_end=1224
  _globals['_SCALARFIELD']._serialized_start=1227
  _globals['_SCALARFIELD']._serialized_end=1737
  _globals['_VECTORFIELD']._serialized_start=1740
  _globals['_VECTORFIELD']._serialized_end=1909
  _globals['_FIELDDATA']._serialized_start=1912
  _globals['_FIELDDATA']._serialized_end=2141
  _globals['_IDS']._serialized_start=2143
  _globals['_IDS']._serialized_end=2262
  _globals['_SEARCHRESULTDATA']._serialized_start=2265
  _globals['_SEARCHRESULTDATA']._serialized_end=2527
# @@protoc_insertion_point(module_scope)
