import logging
from typing import Any
from unittest.mock import patch

import grpc
import pytest
from pymilvus import MilvusException
from pymilvus.client.abstract import AnnSearchRequest, MutationResult, WeightedRanker
from pymilvus.client.asynch import MutationFuture, SearchFuture
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.search_result import SearchResult
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2
from pymilvus.orm.types import DataType

log = logging.getLogger(__name__)

descriptor = milvus_pb2.DESCRIPTOR.services_by_name["MilvusService"]


class TestGrpcHandlerInsertOperations:
    def test_insert_rows(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        # Mock the schema
        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [
                    {"name": "id", "type": DataType.INT64},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}
                ],
                "enable_dynamic_field": False,
                "update_timestamp": 0
            }

            entities = [
                {"id": 1, "vector": [0.1] * 128},
                {"id": 2, "vector": [0.2] * 128}
            ]

            insert_future = client_thread.submit(
                handler.insert_rows,
                collection_name="test_collection",
                entities=entities,
                timeout=10
            )

            (invocation_metadata, request, rpc) = channel.take_unary_unary(
                descriptor.methods_by_name["Insert"]
            )
            rpc.send_initial_metadata(())

            expected_result = milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2])),
                insert_cnt=2,
                timestamp=100
            )
            rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

            result = insert_future.result()
            log.warning(f"result = {result}, type={type(result)}")
            assert isinstance(result, MutationResult)

    def test_insert_rows_single_entity(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [
                    {"name": "id", "type": DataType.INT64}
                ],
                "enable_dynamic_field": False,
                "update_timestamp": 0
            }

            entity = {"id": 1}

            insert_future = client_thread.submit(
                handler.insert_rows,
                collection_name="test_collection",
                entities=entity,  # Single dict instead of list
                timeout=10
            )

            (invocation_metadata, request, rpc) = channel.take_unary_unary(
                descriptor.methods_by_name["Insert"]
            )
            rpc.send_initial_metadata(())

            expected_result = milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
                insert_cnt=1,
                timestamp=100
            )
            rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

            result = insert_future.result()
            assert isinstance(result, MutationResult)

    def test_insert_rows_schema_mismatch(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [
                    {"name": "id", "type": DataType.INT64}
                ],
            }

            entities = [{"id": 1}]

            insert_future = client_thread.submit(
                handler.insert_rows,
                collection_name="test_collection",
                entities=entities,
                timeout=10
            )

            (invocation_metadata, request, rpc) = channel.take_unary_unary(
                descriptor.methods_by_name["Insert"]
            )
            rpc.send_initial_metadata(())

            # Return schema mismatch error
            expected_result = milvus_pb2.MutationResult(
                status=common_pb2.Status(
                    error_code=common_pb2.SchemaMismatch,
                    reason="Schema mismatch"
                )
            )
            rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

            # Should trigger recursive call with updated schema
            (invocation_metadata2, request2, rpc2) = channel.take_unary_unary(
                descriptor.methods_by_name["Insert"]
            )
            rpc2.send_initial_metadata(())

            expected_result2 = milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
                insert_cnt=1,
                timestamp=100
            )
            rpc2.terminate(expected_result2, (), grpc.StatusCode.OK, "")

            result = insert_future.result()
            assert isinstance(result, MutationResult)


class TestGrpcHandlerDeleteAndUpsertOperations:
    def test_delete(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        delete_future = client_thread.submit(
            handler.delete,
            collection_name="test_collection",
            expression="id in [1, 2, 3]",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Delete"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.MutationResult(
            status=common_pb2.Status(code=0),
            delete_cnt=3,
            timestamp=100
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = delete_future.result()
        assert isinstance(result, MutationResult)

    def test_delete_with_partition(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        delete_future = client_thread.submit(
            handler.delete,
            collection_name="test_collection",
            expression="id > 10",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Delete"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.MutationResult(
            status=common_pb2.Status(code=0),
            delete_cnt=5,
            timestamp=100
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = delete_future.result()
        assert isinstance(result, MutationResult)

    def test_delete_async(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        delete_future = client_thread.submit(
            handler.delete,
            collection_name="test_collection",
            expression="id == 1",
            timeout=10,
            _async=True
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Delete"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.MutationResult(
            status=common_pb2.Status(code=0),
            delete_cnt=1,
            timestamp=100
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = delete_future.result()
        assert isinstance(result, MutationFuture)

    def test_upsert_rows(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [
                    {"name": "id", "type": DataType.INT64}
                ],
                "enable_dynamic_field": False,
                "update_timestamp": 0
            }

            entities = [{"id": 1}, {"id": 2}]

            upsert_future = client_thread.submit(
                handler.upsert_rows,
                collection_name="test_collection",
                entities=entities,
                timeout=10
            )

            (invocation_metadata, request, rpc) = channel.take_unary_unary(
                descriptor.methods_by_name["Upsert"]
            )
            rpc.send_initial_metadata(())

            expected_result = milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2])),
                upsert_cnt=2,
                timestamp=100
            )
            rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

            result = upsert_future.result()
            assert isinstance(result, MutationResult)

    def test_upsert_rows_single_entity(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [
                    {"name": "id", "type": DataType.INT64}
                ],
                "enable_dynamic_field": False,
                "update_timestamp": 0
            }

            entity = {"id": 1}  # Single dict

            upsert_future = client_thread.submit(
                handler.upsert_rows,
                collection_name="test_collection",
                entities=entity,
                timeout=10
            )

            (invocation_metadata, request, rpc) = channel.take_unary_unary(
                descriptor.methods_by_name["Upsert"]
            )
            rpc.send_initial_metadata(())

            expected_result = milvus_pb2.MutationResult(
                status=common_pb2.Status(code=0),
                IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
                upsert_cnt=1,
                timestamp=100
            )
            rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

            result = upsert_future.result()
            assert isinstance(result, MutationResult)


class TestGrpcHandlerSearchOperations:

    def test_search(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        search_data = [[0.1] * 128]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        search_future = client_thread.submit(
            handler.search,
            collection_name="test_collection",
            data=search_data,
            anns_field="vector",
            param=search_params,
            limit=10,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Search"]
        )
        rpc.send_initial_metadata(())

        # Create search results
        results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=1,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            scores=[0.5],
            topks=[1]
        )

        expected_result = milvus_pb2.SearchResults(
            status=common_pb2.Status(code=0),
            results=results,
            session_ts=100
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = search_future.result()
        assert isinstance(result, SearchResult)

    def test_search_with_expression(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        search_data = [[0.1] * 128]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        search_future = client_thread.submit(
            handler.search,
            collection_name="test_collection",
            data=search_data,
            anns_field="vector",
            param=search_params,
            limit=10,
            expression="id > 100",
            partition_names=["partition1"],
            output_fields=["id", "value"],
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Search"]
        )
        rpc.send_initial_metadata(())

        results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=1,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[101])),
            scores=[0.3],
            topks=[1]
        )

        expected_result = milvus_pb2.SearchResults(
            status=common_pb2.Status(code=0),
            results=results
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = search_future.result()
        assert isinstance(result, SearchResult)

    def test_search_async(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        search_data = [[0.1] * 128]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        search_future = client_thread.submit(
            handler.search,
            collection_name="test_collection",
            data=search_data,
            anns_field="vector",
            param=search_params,
            limit=10,
            timeout=10,
            _async=True
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["Search"]
        )
        rpc.send_initial_metadata(())

        results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=1,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            scores=[0.5],
            topks=[1]
        )

        expected_result = milvus_pb2.SearchResults(
            status=common_pb2.Status(code=0),
            results=results
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = search_future.result()
        assert isinstance(result, SearchFuture)

    def test_hybrid_search(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        # Create multiple search requests
        req1 = AnnSearchRequest(
            data=[[0.1] * 128],
            anns_field="vector1",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10
        )
        req2 = AnnSearchRequest(
            data=[[0.2] * 128],
            anns_field="vector2",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=10
        )

        rerank = WeightedRanker(0.5, 0.5)

        hybrid_future = client_thread.submit(
            handler.hybrid_search,
            collection_name="test_collection",
            reqs=[req1, req2],
            rerank=rerank,
            limit=10,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["HybridSearch"]
        )
        rpc.send_initial_metadata(())

        results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=2,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2])),
            scores=[0.9, 0.8],
            topks=[2]
        )

        expected_result = milvus_pb2.SearchResults(
            status=common_pb2.Status(code=0),
            results=results
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = hybrid_future.result()
        assert isinstance(result, SearchResult)


class TestGrpcHandlerSegmentAndAliasOperations:

    def test_get_query_segment_info(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        info_future = client_thread.submit(
            handler.get_query_segment_info,
            collection_name="test_collection",
            timeout=30
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetQuerySegmentInfo"]
        )
        rpc.send_initial_metadata(())

        # Create segment info
        segment_info = milvus_pb2.QuerySegmentInfo(
            segmentID=1001,
            collectionID=100,
            partitionID=10,
            num_rows=1000,
            state=common_pb2.SegmentState.Sealed
        )

        expected_result = milvus_pb2.GetQuerySegmentInfoResponse(
            status=common_pb2.Status(code=0),
            infos=[segment_info]
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = info_future.result()
        assert result == [segment_info]

    def test_get_query_segment_info_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        info_future = client_thread.submit(
            handler.get_query_segment_info,
            collection_name="test_collection",
            timeout=30
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetQuerySegmentInfo"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.GetQuerySegmentInfoResponse(
            status=common_pb2.Status(code=1, reason="Collection not found")
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            info_future.result()

    def test_create_alias(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        alias_future = client_thread.submit(
            handler.create_alias,
            collection_name="test_collection",
            alias="test_alias",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreateAlias"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = alias_future.result()
        assert result is None

    def test_create_alias_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        alias_future = client_thread.submit(
            handler.create_alias,
            collection_name="test_collection",
            alias="test_alias",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreateAlias"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=1, reason="Alias already exists")
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            alias_future.result()


class TestGrpcHandlerHelperMethods:
    def test_get_info(self) -> None:
        handler = GrpcHandler(channel=None)

        # Test with schema provided
        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vector", "type": DataType.FLOAT_VECTOR}
            ],
            "enable_dynamic_field": True
        }

        fields_info, enable_dynamic = handler._get_info("test_collection", schema=schema)

        assert fields_info == schema["fields"]
        assert enable_dynamic is True

    def test_get_info_without_schema(self) -> None:
        handler = GrpcHandler(channel=None)

        with patch.object(handler, 'describe_collection') as mock_describe:
            mock_describe.return_value = {
                "fields": [{"name": "id", "type": DataType.INT64}],
                "enable_dynamic_field": False
            }

            fields_info, enable_dynamic = handler._get_info("test_collection")

            assert fields_info == [{"name": "id", "type": DataType.INT64}]
            assert enable_dynamic is False

    def test_get_schema_from_cache_or_remote_cached(self) -> None:
        handler = GrpcHandler(channel=None)

        # Add to cache
        cached_schema = {
            "fields": [{"name": "id", "type": DataType.INT64}],
            "update_timestamp": 100
        }
        handler.schema_cache["test_collection"] = {
            "schema": cached_schema,
            "schema_timestamp": 100
        }

        schema, timestamp = handler._get_schema_from_cache_or_remote("test_collection")

        assert schema == cached_schema
        assert timestamp == 100

    def test_get_schema_from_cache_or_remote_not_cached(self) -> None:
        handler = GrpcHandler(channel=None)

        with patch.object(handler, 'describe_collection') as mock_describe:
            remote_schema = {
                "fields": [{"name": "id", "type": DataType.INT64}],
                "update_timestamp": 200
            }
            mock_describe.return_value = remote_schema

            schema, timestamp = handler._get_schema_from_cache_or_remote("test_collection")

            assert schema == remote_schema
            assert timestamp == 200
            # Check it was cached
            assert handler.schema_cache["test_collection"]["schema"] == remote_schema
            assert handler.schema_cache["test_collection"]["schema_timestamp"] == 200
