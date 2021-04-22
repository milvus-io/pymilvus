# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: schema.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='schema.proto',
  package='milvus.proto.schema',
  syntax='proto3',
  serialized_options=_b('Z@github.com/zilliztech/milvus-distributed/internal/proto/schemapb'),
  serialized_pb=_b('\n\x0cschema.proto\x12\x13milvus.proto.schema\x1a\x0c\x63ommon.proto\"\xfc\x01\n\x0b\x46ieldSchema\x12\x0f\n\x07\x66ieldID\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x16\n\x0eis_primary_key\x18\x03 \x01(\x08\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x30\n\tdata_type\x18\x05 \x01(\x0e\x32\x1d.milvus.proto.schema.DataType\x12\x36\n\x0btype_params\x18\x06 \x03(\x0b\x32!.milvus.proto.common.KeyValuePair\x12\x37\n\x0cindex_params\x18\x07 \x03(\x0b\x32!.milvus.proto.common.KeyValuePair\"w\n\x10\x43ollectionSchema\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x0e\n\x06\x61utoID\x18\x03 \x01(\x08\x12\x30\n\x06\x66ields\x18\x04 \x03(\x0b\x32 .milvus.proto.schema.FieldSchema*\x8f\x01\n\x08\x44\x61taType\x12\x08\n\x04None\x10\x00\x12\x08\n\x04\x42ool\x10\x01\x12\x08\n\x04Int8\x10\x02\x12\t\n\x05Int16\x10\x03\x12\t\n\x05Int32\x10\x04\x12\t\n\x05Int64\x10\x05\x12\t\n\x05\x46loat\x10\n\x12\n\n\x06\x44ouble\x10\x0b\x12\n\n\x06String\x10\x14\x12\x10\n\x0c\x42inaryVector\x10\x64\x12\x0f\n\x0b\x46loatVector\x10\x65\x42\x42Z@github.com/zilliztech/milvus-distributed/internal/proto/schemapbb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])

_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='milvus.proto.schema.DataType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='None', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Bool', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Int8', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Int16', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Int32', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Int64', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Float', index=6, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Double', index=7, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='String', index=8, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BinaryVector', index=9, number=100,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FloatVector', index=10, number=101,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=428,
  serialized_end=571,
)
_sym_db.RegisterEnumDescriptor(_DATATYPE)

DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
globals()['None'] = 0
Bool = 1
Int8 = 2
Int16 = 3
Int32 = 4
Int64 = 5
Float = 10
Double = 11
String = 20
BinaryVector = 100
FloatVector = 101



_FIELDSCHEMA = _descriptor.Descriptor(
  name='FieldSchema',
  full_name='milvus.proto.schema.FieldSchema',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='fieldID', full_name='milvus.proto.schema.FieldSchema.fieldID', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='milvus.proto.schema.FieldSchema.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_primary_key', full_name='milvus.proto.schema.FieldSchema.is_primary_key', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='milvus.proto.schema.FieldSchema.description', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_type', full_name='milvus.proto.schema.FieldSchema.data_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type_params', full_name='milvus.proto.schema.FieldSchema.type_params', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index_params', full_name='milvus.proto.schema.FieldSchema.index_params', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=52,
  serialized_end=304,
)


_COLLECTIONSCHEMA = _descriptor.Descriptor(
  name='CollectionSchema',
  full_name='milvus.proto.schema.CollectionSchema',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='milvus.proto.schema.CollectionSchema.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='milvus.proto.schema.CollectionSchema.description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='autoID', full_name='milvus.proto.schema.CollectionSchema.autoID', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fields', full_name='milvus.proto.schema.CollectionSchema.fields', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=306,
  serialized_end=425,
)

_FIELDSCHEMA.fields_by_name['data_type'].enum_type = _DATATYPE
_FIELDSCHEMA.fields_by_name['type_params'].message_type = common__pb2._KEYVALUEPAIR
_FIELDSCHEMA.fields_by_name['index_params'].message_type = common__pb2._KEYVALUEPAIR
_COLLECTIONSCHEMA.fields_by_name['fields'].message_type = _FIELDSCHEMA
DESCRIPTOR.message_types_by_name['FieldSchema'] = _FIELDSCHEMA
DESCRIPTOR.message_types_by_name['CollectionSchema'] = _COLLECTIONSCHEMA
DESCRIPTOR.enum_types_by_name['DataType'] = _DATATYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FieldSchema = _reflection.GeneratedProtocolMessageType('FieldSchema', (_message.Message,), {
  'DESCRIPTOR' : _FIELDSCHEMA,
  '__module__' : 'schema_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.schema.FieldSchema)
  })
_sym_db.RegisterMessage(FieldSchema)

CollectionSchema = _reflection.GeneratedProtocolMessageType('CollectionSchema', (_message.Message,), {
  'DESCRIPTOR' : _COLLECTIONSCHEMA,
  '__module__' : 'schema_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.schema.CollectionSchema)
  })
_sym_db.RegisterMessage(CollectionSchema)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
