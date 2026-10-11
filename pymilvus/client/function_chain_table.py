"""Independent POST table results; Arrow is an optional, lazily imported dependency."""

import importlib
from typing import Any, Dict, List

from pymilvus.exceptions import MilvusException, ParamError


def require_arrow():
    try:
        return importlib.import_module("pyarrow")
    except ImportError as error:
        raise ParamError(message="POST TABLE requires pyarrow; install pymilvus[arrow]") from error


def _validate_struct_names(arrow: Any, datatype: Any) -> None:
    if arrow.types.is_struct(datatype):
        names = [field.name for field in datatype]
        if len(set(names)) != len(names):
            raise MilvusException(message="POST table Struct member names must be unique")
        for field in datatype:
            _validate_struct_names(arrow, field.type)
    elif (
        arrow.types.is_list(datatype)
        or arrow.types.is_large_list(datatype)
        or arrow.types.is_fixed_size_list(datatype)
    ):
        _validate_struct_names(arrow, datatype.value_type)
    elif arrow.types.is_map(datatype):
        _validate_struct_names(arrow, datatype.key_type)
        _validate_struct_names(arrow, datatype.item_type)
    elif arrow.types.is_dictionary(datatype):
        _validate_struct_names(arrow, datatype.value_type)
    elif arrow.types.is_union(datatype):
        for field in datatype:
            _validate_struct_names(arrow, field.type)


class FunctionChainTableResult:
    """A POST table with its runtime schema, independent of search hits/NQ.

    ``rows``/``to_rows()`` materialize Python dictionaries. ``to_arrow()`` retains
    nested Arrow types and schema metadata without coercing to collection fields.
    """

    def __init__(
        self, result: Any, *, status: Any = None, collection_name: str = "", session_ts: int = 0
    ):
        if result.row_count < 0 or not result.arrow_ipc:
            raise MilvusException(message="Invalid POST table row count or missing Arrow schema")
        arrow = require_arrow()
        source = arrow.BufferReader(result.arrow_ipc)
        try:
            with arrow.ipc.open_stream(source) as reader:
                table = reader.read_all()
            # IPC reading checks framing, not all array contents. Validate before
            # publishing schema/rows so malformed UTF-8, offsets or dictionaries
            # cannot become a partially usable result object.
            table.validate(full=True)
            if source.tell() != len(result.arrow_ipc):
                raise MilvusException(message="POST table Arrow stream contains trailing data")
            if table.num_rows != result.row_count:
                raise MilvusException(message="POST table row_count differs from Arrow rows")
            if len(set(table.schema.names)) != len(table.schema.names):
                raise MilvusException(message="POST table column names must be unique")
            for field, column in zip(table.schema, table.columns):
                if not field.nullable and column.null_count:
                    raise MilvusException(
                        message="POST table non-nullable column contains null values"
                    )
                _validate_struct_names(arrow, field.type)
        except MilvusException:
            raise
        except Exception as error:
            raise MilvusException(message="Invalid POST table Arrow IPC stream") from error
        self._table = table
        self._arrow_ipc = result.arrow_ipc
        self.extra = dict(status.extra_info) if status is not None else {}
        self.collection_name = collection_name
        self.session_ts = session_ts

    @property
    def schema(self):
        return self._table.schema

    @property
    def row_count(self) -> int:
        return self._table.num_rows

    @property
    def num_rows(self) -> int:
        return self.row_count

    @property
    def columns(self) -> List[str]:
        return self._table.column_names

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return self.to_rows()

    @property
    def arrow_ipc(self) -> bytes:
        return self._arrow_ipc

    def to_arrow(self):
        return self._table

    def to_rows(self) -> List[Dict[str, Any]]:
        return self._table.to_pylist()

    def to_pandas(self, **kwargs):
        return self._table.to_pandas(**kwargs)

    def __len__(self) -> int:
        return self.row_count

    def __repr__(self) -> str:
        return f"FunctionChainTableResult(rows={self.row_count}, schema={self.schema})"


TableResult = FunctionChainTableResult


def parse_table_response(response: Any, *, expected: bool):
    """Require a TABLE acknowledgment even if an older Proxy returned valid hits."""
    present = "post_process_result" in response.DESCRIPTOR.fields_by_name and response.HasField(
        "post_process_result"
    )
    if expected:
        if not present:
            raise MilvusException(message="POST TABLE response is missing post_process_result")
        # A new independent table must not be combined with ordinary hit/field
        # payload. Do not silently pick one of two conflicting response shapes.
        fields = response.DESCRIPTOR.fields_by_name
        if "results" in fields and response.HasField("results"):
            raise MilvusException(
                message="POST TABLE response also contains ordinary search results"
            )
        if "fields_data" in fields and response.fields_data:
            raise MilvusException(message="POST TABLE response also contains ordinary query fields")
        return FunctionChainTableResult(
            response.post_process_result,
            status=response.status,
            collection_name=response.collection_name,
            session_ts=response.session_ts,
        )
    if present:
        raise MilvusException(message="Unexpected POST TABLE response for an ordinary request")
    return None
