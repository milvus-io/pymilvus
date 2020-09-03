#!/bin/bash


python -m grpc_tools.protoc -I . --python_out=./gen --grpc_python_out=./gen status.proto


python -m grpc_tools.protoc -I . --python_out=./gen --grpc_python_out=./gen milvus.proto