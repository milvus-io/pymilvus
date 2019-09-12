import os
import sys
import pathlib


root_path = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(root_path))

from milvus.grpc_gen import status_pb2
from milvus.grpc_gen import status_pb2_grpc
from milvus.grpc_gen import milvus_pb2
from milvus.grpc_gen import milvus_pb2_grpc


__all__ = ['status_pb2', 'status_pb2_grpc', 'milvus_pb2', 'milvus_pb2_grpc']
