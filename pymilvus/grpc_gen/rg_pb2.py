# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rg.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08rg.proto\x12\x0fmilvus.proto.rg\"&\n\x12ResourceGroupLimit\x12\x10\n\x08node_num\x18\x01 \x01(\x05\"/\n\x15ResourceGroupTransfer\x12\x16\n\x0eresource_group\x18\x01 \x01(\t\"\xfd\x01\n\x13ResourceGroupConfig\x12\x35\n\x08requests\x18\x01 \x01(\x0b\x32#.milvus.proto.rg.ResourceGroupLimit\x12\x33\n\x06limits\x18\x02 \x01(\x0b\x32#.milvus.proto.rg.ResourceGroupLimit\x12=\n\rtransfer_from\x18\x03 \x03(\x0b\x32&.milvus.proto.rg.ResourceGroupTransfer\x12;\n\x0btransfer_to\x18\x04 \x03(\x0b\x32&.milvus.proto.rg.ResourceGroupTransferBp\n\x0eio.milvus.grpcB\x12ResourceGroupProtoP\x01Z0github.com/milvus-io/milvus-proto/go-api/v2/rgpb\xa0\x01\x01\xaa\x02\x12Milvus.Client.Grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rg_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\016io.milvus.grpcB\022ResourceGroupProtoP\001Z0github.com/milvus-io/milvus-proto/go-api/v2/rgpb\240\001\001\252\002\022Milvus.Client.Grpc'
  _globals['_RESOURCEGROUPLIMIT']._serialized_start=29
  _globals['_RESOURCEGROUPLIMIT']._serialized_end=67
  _globals['_RESOURCEGROUPTRANSFER']._serialized_start=69
  _globals['_RESOURCEGROUPTRANSFER']._serialized_end=116
  _globals['_RESOURCEGROUPCONFIG']._serialized_start=119
  _globals['_RESOURCEGROUPCONFIG']._serialized_end=372
# @@protoc_insertion_point(module_scope)
