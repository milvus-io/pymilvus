"""Exercise public clients over a local grpcio transport using gzip."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager

import grpc
import pytest
from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import milvus_pb2, milvus_pb2_grpc

COLLECTION = "compressible" * 12
VERSION = "version" * 8000


class CompressionService(milvus_pb2_grpc.MilvusServiceServicer):
    def Connect(self, request, context):
        return milvus_pb2.ConnectResponse(identifier=27)

    def DescribeCollection(self, request, context):
        if request.collection_name != COLLECTION:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "incorrect collection name")
        return milvus_pb2.DescribeCollectionResponse()

    def GetVersion(self, request, context):
        return milvus_pb2.GetVersionResponse(version=VERSION)


@contextmanager
def endpoint(algorithm):
    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(
            executor,
            compression=algorithm,
            options=[("grpc.compression_enabled_algorithms_bitset", 1 | (1 << algorithm))],
        )
        milvus_pb2_grpc.add_MilvusServiceServicer_to_server(CompressionService(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.stop(0).wait()


def test_gzip_sync():
    with endpoint(grpc.Compression.Gzip) as uri:
        with closing(MilvusClient(uri, grpc_compression="gzip", timeout=5)) as client:
            assert client.has_collection(COLLECTION, timeout=5)
            assert client.get_server_version(timeout=5) == VERSION


@pytest.mark.asyncio
async def test_gzip_async():
    with endpoint(grpc.Compression.Gzip) as uri:
        async with AsyncMilvusClient(uri, grpc_compression="gzip", timeout=5) as client:
            assert await client.has_collection(COLLECTION, timeout=5)
            assert await client.get_server_version(timeout=5) == VERSION


def test_gzip_sync_with_identity_only_peer():
    # The gzip client must send gzip even with a default connection already open.
    with endpoint(grpc.Compression.NoCompression) as uri:
        with closing(MilvusClient(uri, timeout=5)) as default:
            assert default.get_server_version(timeout=5) == VERSION
            with pytest.raises(MilvusException) as exc:
                MilvusClient(uri, grpc_compression="gzip", timeout=5)
            assert exc.value.__cause__.code() == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_gzip_async_with_identity_only_peer():
    with endpoint(grpc.Compression.NoCompression) as uri:
        async with AsyncMilvusClient(uri, timeout=5) as default:
            assert await default.get_server_version(timeout=5) == VERSION
            with pytest.raises(MilvusException) as exc:
                async with AsyncMilvusClient(uri, grpc_compression="gzip", timeout=5):
                    pass
            assert exc.value.__cause__.code() == grpc.StatusCode.UNIMPLEMENTED
