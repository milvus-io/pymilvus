#!/bin/bash

OUTDIR=.
PROTO_DIR="milvus-proto/proto"

python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} ${PROTO_DIR}/common.proto
python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} ${PROTO_DIR}/schema.proto
python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} ${PROTO_DIR}/feder.proto
python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} ${PROTO_DIR}/msg.proto
python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} ${PROTO_DIR}/rg.proto
python3 -m grpc_tools.protoc -I ${PROTO_DIR} --python_out=${OUTDIR} --pyi_out=${OUTDIR} --grpc_python_out=${OUTDIR}  ${PROTO_DIR}/milvus.proto

if [[ $(uname -s) == "Darwin" ]]; then
    if ! brew --prefix --installed gnu-sed >/dev/null 2>&1; then
        brew install gnu-sed
    fi
    export PATH="/usr/local/opt/gsed/libexec/gnubin:$PATH"
fi

sed -i 's/import common_pb2 as common__pb2/from . import common_pb2 as common__pb2/' $OUTDIR/*py
sed -i 's/import schema_pb2 as schema__pb2/from . import schema_pb2 as schema__pb2/' $OUTDIR/*py
sed -i 's/import milvus_pb2 as milvus__pb2/from . import milvus_pb2 as milvus__pb2/' $OUTDIR/*py
sed -i 's/import feder_pb2 as feder__pb2/from . import feder_pb2 as feder__pb2/' $OUTDIR/*py
sed -i 's/import msg_pb2 as msg__pb2/from . import msg_pb2 as msg__pb2/' $OUTDIR/*py
sed -i 's/import rg_pb2 as rg__pb2/from . import rg_pb2 as rg__pb2/' $OUTDIR/*py

sed -i 's/import common_pb2 as _common_pb2/from . import common_pb2 as _common_pb2/' $OUTDIR/*pyi
sed -i 's/import schema_pb2 as _schema_pb2/from . import schema_pb2 as _schema_pb2/' $OUTDIR/*pyi
sed -i 's/import milvus_pb2 as _milvus_pb2/from . import milvus_pb2 as _milvus_pb2/' $OUTDIR/*pyi
sed -i 's/import feder_pb2 as _feder_pb2/from . import feder_pb2 as _feder_pb2/' $OUTDIR/*pyi
sed -i 's/import msg_pb2 as _msg_pb2/from . import msg_pb2 as _msg_pb2/' $OUTDIR/*pyi
sed -i 's/import rg_pb2 as _rg_pb2/from . import rg_pb2 as _rg_pb2/' $OUTDIR/*pyi

# WORKAROUND: Remove _registered_method parameter for compatibility with grpcio-testing
# This is a known issue where grpcio-testing's TestingChannel doesn't support the
# _registered_method parameter that was added in grpcio 1.65.0+
# The grpcio-testing package's TestingChannel.unary_unary() method signature hasn't been
# updated to accept this parameter, even though the main grpcio package generates it.
# TODO: Remove this workaround when grpcio-testing is updated to support this parameter
sed -i 's/, _registered_method=True)/)/g' $OUTDIR/milvus_pb2_grpc.py
sed -i 's/_registered_method=True)/)/g' $OUTDIR/milvus_pb2_grpc.py

echo "Success to generate the python proto files."
