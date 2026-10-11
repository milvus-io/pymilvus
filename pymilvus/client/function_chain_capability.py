"""Capability checks for stages that older servers can silently ignore."""

from google.protobuf.message import Message

from pymilvus.exceptions import IngestionFunctionChainError, MilvusException
from pymilvus.function_chain import FunctionChain
from pymilvus.grpc_gen import schema_pb2

from .abstract import MutationResult
from .utils import check_status

TRANSFORM_VERSION_KEY = "function_chain_transform_version"
POST_TABLE_VERSION_KEY = "function_chain_post_table_version"


def requires_table_capability(request: Message) -> bool:
    return any(
        chain.stage == schema_pb2.FunctionChainStagePostProcess
        and chain.post_process_options.result_mode == schema_pb2.FunctionChainResultModeTable
        for chain in getattr(request, "function_chains", ())
    )


def requires_transform_capability(request: Message) -> bool:
    if "schema" in request.DESCRIPTOR.fields_by_name:
        schema = schema_pb2.CollectionSchema.FromString(request.schema)
        return bool(schema.function_chains)
    for chain in getattr(request, "function_chains", ()):
        if chain.stage in (
            schema_pb2.FunctionChainStageIngestion,
            schema_pb2.FunctionChainStagePreProcess,
            schema_pb2.FunctionChainStagePostProcess,
        ) or any(op.op in ("flat_map", "explode", "select") for op in chain.ops):
            return True
    return any(requires_transform_capability(sub) for sub in getattr(request, "requests", ()))


def ingestion_binding(schema: dict) -> schema_pb2.CollectionSchema:
    return schema_pb2.CollectionSchema(
        function_chains=[
            (
                chain.to_proto()
                if isinstance(chain, FunctionChain)
                else FunctionChain.from_dict(chain).to_proto()
            )
            for chain in schema.get("function_chains", [])
        ]
    )


def check_ingestion_response(response: Message) -> None:
    try:
        check_status(response.status)
    except MilvusException as error:
        if response.HasField("ingestion_result") is not True:
            raise
        raise IngestionFunctionChainError(
            error.code,
            error.message,
            error.compatible_code,
            is_input_error=error.is_input_error,
            mutation_result=MutationResult(response),
            unknown_outcome=response.ingestion_result.outcome_unknown,
        ) from error
