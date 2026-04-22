# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Field-level partial-update helpers.

These helpers wrap :class:`schema_pb2.FieldPartialUpdateOp` so callers do not
need to import the generated protobuf module directly. ``FieldOp.array_append``
and ``FieldOp.array_remove`` return ready-to-use proto messages that can be
attached to a field during ``MilvusClient.upsert``.
"""

from typing import Mapping, Optional, Union

from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2

__all__ = ["FieldOp", "FieldOpType", "normalize_field_ops"]

# Re-exported for callers that want to compare against the low-level enum
# (e.g. in tests) without reaching into the generated module.
FieldOpType = schema_pb2.FieldPartialUpdateOp.OpType

# FieldOpsInput describes the accepted input shape of the ``field_ops`` kwarg
# across the public upsert API. The value per field may be:
#
#   * a :class:`schema_pb2.FieldPartialUpdateOp` message directly;
#   * a :class:`FieldOpType` enum value;
#   * a string alias like ``"array_append"`` / ``"array_remove"``.
FieldOpsInput = Optional[Mapping[str, Union[schema_pb2.FieldPartialUpdateOp, "FieldOpType", str]]]


class FieldOp:
    """Factory for building :class:`FieldPartialUpdateOp` messages.

    Prefer these factories over constructing the proto message directly so
    the surface stays stable if the proto changes.
    """

    @staticmethod
    def replace() -> schema_pb2.FieldPartialUpdateOp:
        """Return an explicit REPLACE op — equivalent to omitting the op.

        Exposed for completeness; typical callers just leave ``field_ops``
        unset for the REPLACE (default) behavior.
        """
        return schema_pb2.FieldPartialUpdateOp(op=FieldOpType.REPLACE)

    @staticmethod
    def array_append() -> schema_pb2.FieldPartialUpdateOp:
        """Return an ARRAY_APPEND op, appending the payload to the base row."""
        return schema_pb2.FieldPartialUpdateOp(op=FieldOpType.ARRAY_APPEND)

    @staticmethod
    def array_remove() -> schema_pb2.FieldPartialUpdateOp:
        """Return an ARRAY_REMOVE op, deleting every matching element."""
        return schema_pb2.FieldPartialUpdateOp(op=FieldOpType.ARRAY_REMOVE)


# Map from string alias to the concrete enum value. Kept out of the class so
# callers can extend it in the future without touching ``FieldOp``.
_STRING_ALIASES = {
    "replace": FieldOpType.REPLACE,
    "array_append": FieldOpType.ARRAY_APPEND,
    "array_remove": FieldOpType.ARRAY_REMOVE,
}


def normalize_field_ops(
    field_ops: FieldOpsInput,
) -> "dict[str, schema_pb2.FieldPartialUpdateOp]":
    """Coerce the ``field_ops`` kwarg into a dict of proto messages.

    Accepts ``None``, a mapping of field name to proto message / enum / string
    alias, and returns a dict keyed by field name. Entries that resolve to
    REPLACE are dropped, since REPLACE is the default on the wire and would
    otherwise waste bytes.

    Raises :class:`ParamError` for unknown string aliases or unsupported types
    — the goal is to surface user mistakes at request-prep time rather than
    as a cryptic server-side error.
    """
    if not field_ops:
        return {}
    normalized: dict[str, schema_pb2.FieldPartialUpdateOp] = {}
    for field_name, op in field_ops.items():
        resolved = _coerce_single(field_name, op)
        if resolved is None or resolved.op == FieldOpType.REPLACE:
            continue
        normalized[field_name] = resolved
    return normalized


def _coerce_single(
    field_name: str,
    op: Union[schema_pb2.FieldPartialUpdateOp, "FieldOpType", str, None],
) -> Optional[schema_pb2.FieldPartialUpdateOp]:
    if op is None:
        return None
    if isinstance(op, schema_pb2.FieldPartialUpdateOp):
        return op
    if isinstance(op, str):
        alias = op.lower()
        if alias not in _STRING_ALIASES:
            raise ParamError(message=f"unknown field_op alias {op!r} for field {field_name!r}")
        return schema_pb2.FieldPartialUpdateOp(op=_STRING_ALIASES[alias])
    # Fall back: treat it as a raw enum value if it is an int.
    if isinstance(op, int):
        return schema_pb2.FieldPartialUpdateOp(op=op)
    raise ParamError(
        message=f"unsupported field_op type {type(op).__name__} for field {field_name!r}"
    )
