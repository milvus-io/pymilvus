"""Builder primitives for Milvus function chains.

This module provides a small Python DSL for composing function chain expressions
and operations, then serializing them into the protobuf messages expected by
Milvus.

Examples:
    Build a rerank chain that computes a score, sorts by it, and keeps the top
    results::

        from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn

        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="fresh_popular_rerank")
            .map(
                "freshness",
                fn.decay(col("published_at"), function="exp", origin=1700000000, scale=86400),
            )
            .map(
                "$score",
                fn.num_combine(
                    col("$score"),
                    col("freshness"),
                    col("popularity"),
                    mode="weighted",
                    weights=[0.7, 0.2, 0.1],
                ),
            )
            .map("$score", fn.round_decimal(col("$score"), decimal=4))
            .sort(col("$score"), desc=True, tie_break_col=col("$id"))
            .limit(10)
        )
        proto = chain.to_proto()
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2


class FunctionChainStage(IntEnum):
    """Execution stages where a function chain can run.

    Values:
        UNSPECIFIED: Stage is not specified.
        INGESTION: Runs during data ingestion.
        PRE_PROCESS: Runs before the main query/search process.
        L0_RERANK: Runs at the first reranking stage.
        L1_RERANK: Runs at the second reranking stage.
        L2_RERANK: Runs at the third reranking stage.
        POST_PROCESS: Runs after the main query/search process.

    Examples:
        Create a chain for the search reranking stage::

            chain = FunctionChain(FunctionChainStage.L2_RERANK)
    """

    UNSPECIFIED = schema_pb2.FunctionChainStageUnspecified
    INGESTION = schema_pb2.FunctionChainStageIngestion
    PRE_PROCESS = schema_pb2.FunctionChainStagePreProcess
    L0_RERANK = schema_pb2.FunctionChainStageL0Rerank
    L1_RERANK = schema_pb2.FunctionChainStageL1Rerank
    L2_RERANK = schema_pb2.FunctionChainStageL2Rerank
    POST_PROCESS = schema_pb2.FunctionChainStagePostProcess


@dataclass(frozen=True)
class ColumnRef:
    """Reference to a collection field used as a function chain argument.

    Args:
        name: Name of the collection field to reference. Must be a non-empty
            string.

    Raises:
        ParamError: If ``name`` is not a string or is empty.

    Examples:
        Use ``ColumnRef`` directly or create one with :func:`col`::

            score_ref = ColumnRef("score")
            expr = fn.round_decimal(score_ref, decimal=3)
    """

    name: str

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ParamError(
                message=f"Column name must be a string, got {type(self.name).__name__}"
            )
        if not self.name:
            raise ParamError(message="Column name must not be empty")

    def to_proto(self) -> schema_pb2.FunctionChainExprArg:
        return schema_pb2.FunctionChainExprArg(
            column=schema_pb2.FunctionChainColumnArg(name=self.name)
        )


def col(name: str) -> ColumnRef:
    """Create a column reference for use in a function chain expression.

    Args:
        name: Name of the collection field to reference. Must be a non-empty
            string.

    Returns:
        A :class:`ColumnRef` that serializes as a column argument.

    Examples:
        Pass collection fields as expression inputs::

            expr = fn.round_decimal(col("$score"), decimal=3)
    """
    return ColumnRef(name)


FunctionChainArg = Union[ColumnRef, Any]


def _to_param_value(value: Any) -> schema_pb2.FunctionParamValue:
    if value is None:
        raise ParamError(message="Function chain parameters do not support None")

    if isinstance(value, (bool, np.bool_)):
        return schema_pb2.FunctionParamValue(bool_value=bool(value))

    if isinstance(value, (int, np.integer)):
        return schema_pb2.FunctionParamValue(int64_value=int(value))

    if isinstance(value, (float, np.floating)):
        return schema_pb2.FunctionParamValue(double_value=float(value))

    if isinstance(value, str):
        return schema_pb2.FunctionParamValue(string_value=value)

    if isinstance(value, bytes):
        return schema_pb2.FunctionParamValue(bytes_value=value)

    if isinstance(value, (list, tuple)):
        return schema_pb2.FunctionParamValue(
            array_value=schema_pb2.FunctionParamArray(
                values=[_to_param_value(item) for item in value]
            )
        )

    if isinstance(value, Mapping):
        fields = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ParamError(message="Function chain parameter names must be non-empty strings")
            fields[key] = _to_param_value(item)
        return schema_pb2.FunctionParamValue(
            object_value=schema_pb2.FunctionParamObject(fields=fields)
        )

    raise ParamError(message=f"Unsupported function chain parameter type: {type(value).__name__}")


def _to_expr_arg(value: FunctionChainArg) -> schema_pb2.FunctionChainExprArg:
    if isinstance(value, ColumnRef):
        return value.to_proto()
    return schema_pb2.FunctionChainExprArg(literal=_to_param_value(value))


def _copy_param_map(target: Any, params: Mapping[str, Any]) -> None:
    for key, value in params.items():
        if not isinstance(key, str) or not key:
            raise ParamError(message="Function chain parameter names must be non-empty strings")
        target[key].CopyFrom(_to_param_value(value))


def _column_name(value: Union[str, ColumnRef], param_name: str) -> str:
    if isinstance(value, ColumnRef):
        return value.name
    if isinstance(value, str) and value:
        return value
    raise ParamError(message=f"{param_name} must be a non-empty string or ColumnRef")


@dataclass(frozen=True)
class FunctionChainExpr:
    """Function invocation expression used by a function chain operation.

    Args:
        name: Name of the function to invoke. Must be a non-empty string.
        args: Positional expression arguments. Each argument can be a
            :class:`ColumnRef` for field input or a supported Python literal.
        params: Keyword-style function parameters. Keys must be non-empty
            strings, and values must be supported Python literals.

    Raises:
        ParamError: If ``name`` is empty or ``params`` is not a mapping.

    Examples:
        Build an expression that combines two score columns::

            expr = FunctionChainExpr(
                "num_combine",
                args=(col("$score"), col("freshness")),
                params={"mode": "weighted", "weights": [0.8, 0.2]},
            )
    """

    name: str
    args: Tuple[FunctionChainArg, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ParamError(message="Function chain expression name must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise ParamError(message="Function chain expression params must be a mapping")

    def to_proto(self) -> schema_pb2.FunctionChainExpr:
        proto = schema_pb2.FunctionChainExpr(name=self.name)
        proto.args.extend([_to_expr_arg(arg) for arg in self.args])
        _copy_param_map(proto.params, self.params)
        return proto


@dataclass(frozen=True)
class FunctionChainOp:
    """Single operation in a function chain pipeline.

    Args:
        op: Operation name, such as ``"map"``, ``"sort"``, or ``"limit"``.
            Must be a non-empty string.
        expr: Optional expression attached to the operation. ``map`` operations
            use this to compute output values.
        inputs: Input column names consumed by the operation.
        outputs: Output column names produced by the operation.
        params: Operation parameters. Keys must be non-empty strings, and values
            must be supported Python literals.

    Raises:
        ParamError: If ``op`` is empty, ``expr`` is not a
            :class:`FunctionChainExpr`, or ``params`` is not a mapping.

    Examples:
        Create a map operation manually::

            op = FunctionChainOp(
                op="map",
                expr=FunctionChainExpr(
                    "round_decimal",
                    args=(col("$score"),),
                    params={"decimal": 3},
                ),
                outputs=("rounded_score",),
            )
    """

    op: str
    expr: Optional[FunctionChainExpr] = None
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.op, str) or not self.op:
            raise ParamError(message="Function chain op name must be a non-empty string")
        if self.expr is not None and not isinstance(self.expr, FunctionChainExpr):
            raise ParamError(message="Function chain op expr must be a FunctionChainExpr")
        if not isinstance(self.params, Mapping):
            raise ParamError(message="Function chain op params must be a mapping")

    def to_proto(self) -> schema_pb2.FunctionChainOp:
        proto = schema_pb2.FunctionChainOp(op=self.op)
        if self.expr is not None:
            proto.expr.CopyFrom(self.expr.to_proto())
        proto.inputs.extend(self.inputs)
        proto.outputs.extend(self.outputs)
        _copy_param_map(proto.params, self.params)
        return proto


class FunctionChain:
    """Mutable builder for composing and serializing function chain operations.

    Args:
        stage: Execution stage where the chain should run. Accepts a
            :class:`FunctionChainStage` value or an integer value that can be
            converted to one.
        name: Optional chain name. Must be a string.

    Raises:
        ParamError: If ``stage`` is unknown or ``name`` is not a string.

    Examples:
        Build a complete chain with the fluent API::

            chain = (
                FunctionChain(FunctionChainStage.L2_RERANK, name="fresh_popular_rerank")
                .map(
                    "freshness",
                    fn.decay(col("published_at"), function="exp", origin=1700000000, scale=86400),
                )
                .map(
                    "$score",
                    fn.num_combine(
                        col("$score"),
                        col("freshness"),
                        col("popularity"),
                        mode="weighted",
                        weights=[0.7, 0.2, 0.1],
                    ),
                )
                .map("$score", fn.round_decimal(col("$score"), decimal=4))
                .sort(col("$score"), desc=True)
                .limit(10, offset=0)
            )
            proto = chain.to_proto()
    """

    def __init__(self, stage: FunctionChainStage, name: str = ""):
        """Create a function chain builder for the given execution stage.

        Args:
            stage: Execution stage where the chain should run.
            name: Optional chain name.
        """
        try:
            self._stage = FunctionChainStage(stage)
        except ValueError as err:
            raise ParamError(message=f"Unknown function chain stage: {stage}") from err
        if not isinstance(name, str):
            raise ParamError(message="Function chain name must be a string")
        self._name = name
        self._ops: List[FunctionChainOp] = []

    @property
    def name(self) -> str:
        """Name assigned to this function chain.

        Returns:
            The chain name passed to :class:`FunctionChain`.
        """
        return self._name

    @property
    def stage(self) -> FunctionChainStage:
        """Execution stage where this function chain runs.

        Returns:
            The normalized :class:`FunctionChainStage`.
        """
        return self._stage

    @property
    def ops(self) -> Sequence[FunctionChainOp]:
        """Immutable view of operations added to this chain.

        Returns:
            A tuple containing the operations in insertion order.
        """
        return tuple(self._ops)

    def map(self, output: str, expr: FunctionChainExpr) -> "FunctionChain":
        """Append a map operation that writes an expression result to an output field.

        Args:
            output: Name of the output field produced by the expression. Must be
                a non-empty string.
            expr: Function expression to evaluate for each input item.

        Returns:
            ``self`` so calls can be chained fluently.

        Raises:
            ParamError: If ``output`` is empty or ``expr`` is not a
                :class:`FunctionChainExpr`.

        Examples:
            Round a score into a new output field::

                chain.map("rounded_score", fn.round_decimal(col("$score"), decimal=3))
        """
        if not isinstance(output, str) or not output:
            raise ParamError(message="Function chain map output must be a non-empty string")
        if not isinstance(expr, FunctionChainExpr):
            raise ParamError(message="Function chain map expr must be a FunctionChainExpr")
        self._ops.append(FunctionChainOp(op="map", expr=expr, outputs=(output,)))
        return self

    def sort(
        self,
        by: Union[str, ColumnRef],
        desc: bool = True,
        tie_break_col: Optional[Union[str, ColumnRef]] = None,
    ) -> "FunctionChain":
        """Append a sort operation by column, optionally with a tie-break column.

        Args:
            by: Column name or :class:`ColumnRef` used as the primary sort key.
            desc: Whether to sort in descending order. Defaults to ``True``.
            tie_break_col: Optional column name or :class:`ColumnRef` used to
                break ties when primary sort values are equal.

        Returns:
            ``self`` so calls can be chained fluently.

        Raises:
            ParamError: If ``by`` or ``tie_break_col`` is invalid, or ``desc`` is
                not a boolean.

        Examples:
            Sort by score descending, then by document id for stable ordering::

                chain.sort("score", desc=True, tie_break_col="doc_id")
        """
        column = _column_name(by, "sort by")
        if not isinstance(desc, bool):
            raise ParamError(message="Function chain sort desc must be a boolean")
        params: Dict[str, Any] = {"column": column, "desc": desc}
        inputs = [column]
        if tie_break_col is not None:
            tie_break_name = _column_name(tie_break_col, "tie_break_col")
            params["tie_break_col"] = tie_break_name
            inputs.append(tie_break_name)
        self._ops.append(FunctionChainOp(op="sort", inputs=tuple(inputs), params=params))
        return self

    def limit(self, limit: int, offset: int = 0) -> "FunctionChain":
        """Append a limit operation with an optional offset.

        Args:
            limit: Maximum number of items to keep. Must be a positive integer.
            offset: Number of items to skip before applying ``limit``. Must be a
                non-negative integer. Defaults to ``0``.

        Returns:
            ``self`` so calls can be chained fluently.

        Raises:
            ParamError: If ``limit`` is not positive or ``offset`` is negative.

        Examples:
            Keep the first page of ten sorted results::

                chain.limit(10)

            Keep the second page of ten sorted results::

                chain.limit(10, offset=10)
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ParamError(message="Function chain limit must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ParamError(message="Function chain offset must be a non-negative integer")
        self._ops.append(FunctionChainOp(op="limit", params={"limit": limit, "offset": offset}))
        return self

    def to_proto(self) -> schema_pb2.FunctionChain:
        proto = schema_pb2.FunctionChain(name=self._name, stage=int(self._stage))
        proto.ops.extend([op.to_proto() for op in self._ops])
        return proto
