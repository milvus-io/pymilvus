#!/bin/bash
OUTDIR=../

python -m grpc_tools.protoc -I . --python_out=${OUTDIR} common.proto
python -m grpc_tools.protoc -I . --python_out=${OUTDIR} schema.proto
python -m grpc_tools.protoc -I . --python_out=${OUTDIR} --grpc_python_out=${OUTDIR} milvus.proto

sed -i 's/import common_pb2 as common__pb2/from . import common_pb2 as common__pb2/' $OUTDIR/*py
sed -i 's/import schema_pb2 as schema__pb2/from . import schema_pb2 as schema__pb2/' $OUTDIR/*py
sed -i 's/import milvus_pb2 as milvus__pb2/from . import milvus_pb2 as milvus__pb2/' $OUTDIR/*py
