#!/bin/bash

OUTDIR=.
PROTO_DIR="milvus-proto/proto"

python -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR}                              ${PROTO_DIR}/common.proto
python -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR}                              ${PROTO_DIR}/schema.proto
python -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --grpc_python_out=${OUTDIR}  ${PROTO_DIR}/milvus.proto

if [[ $(uname -s) == "Darwin" ]]; then
    if ! brew --prefix --installed gnu-sed >/dev/null 2>&1; then
        brew install gnu-sed
    fi
    export PATH="/usr/local/opt/gsed/libexec/gnubin:$PATH"
fi

sed -i 's/import common_pb2 as common__pb2/from . import common_pb2 as common__pb2/' $OUTDIR/*py
sed -i 's/import schema_pb2 as schema__pb2/from . import schema_pb2 as schema__pb2/' $OUTDIR/*py
sed -i 's/import milvus_pb2 as milvus__pb2/from . import milvus_pb2 as milvus__pb2/' $OUTDIR/*py

echo "Success to generate the python proto files."
