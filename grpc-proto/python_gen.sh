#!/bin/bash

python -m grpc_tools.protoc -I . --python_out=./gen common.proto
python -m grpc_tools.protoc -I . --python_out=./gen schema.proto
python -m grpc_tools.protoc -I . --python_out=./gen --grpc_python_out=./gen milvus.proto
#python -m grpc_tools.protoc -I . --python_out=./gen --grpc_python_out=./gen service.proto
#python -m grpc_tools.protoc -I . --python_out=./gen --grpc_python_out=./gen service_msg.proto
