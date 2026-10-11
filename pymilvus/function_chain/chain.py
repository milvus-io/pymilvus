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
from google.protobuf.json_format import MessageToDict, ParseDict

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
    PRE_INSERT = INGESTION
    PRE_PROCESS = schema_pb2.FunctionChainStagePreProcess
    L0_RERANK = schema_pb2.FunctionChainStageL0Rerank
    L1_RERANK = schema_pb2.FunctionChainStageL1Rerank
    L2_RERANK = schema_pb2.FunctionChainStageL2Rerank
    POST_PROCESS = schema_pb2.FunctionChainStagePostProcess


class FunctionChainResultMode(IntEnum):
    """POST HITS preserves candidates; TABLE returns an independent Arrow table."""

    HITS = schema_pb2.FunctionChainResultModeHits
    TABLE = schema_pb2.FunctionChainResultModeTable


class PostProcessInput(IntEnum):
    """POST input after fusion, or every reduced Hybrid ANN branch."""

    FUSED = schema_pb2.PostProcessInputFused
    ANN_INPUTS = schema_pb2.PostProcessInputAnnInputs


def _post_option(value: Any, enum: Any, label: str):
    if isinstance(value, str):
        try:
            return enum[value.upper()]
        except KeyError as err:
            raise ParamError(message=f"Unknown POST {label}: {value!r}") from err
    if isinstance(value, bool):
        raise ParamError(message=f"Unknown POST {label}: {value!r}")
    try:
        return enum(value)
    except (ValueError, TypeError) as err:
        raise ParamError(message=f"Unknown POST {label}: {value!r}") from err


_SOURCE_COLUMNS = {"$query_index", "$ann_index"}


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
    data_type: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ParamError(
                message=f"Column name must be a string, got {type(self.name).__name__}"
            )
        if not self.name:
            raise ParamError(message="Column name must not be empty")
        if self.data_type is not None:
            try:
                schema_pb2.DataType.Name(int(self.data_type))
            except (TypeError, ValueError) as err:
                raise ParamError(message="Column data_type must be a Milvus DataType") from err

    def to_proto(self) -> schema_pb2.FunctionChainExprArg:
        return schema_pb2.FunctionChainExprArg(
            column=schema_pb2.FunctionChainColumnArg(name=self.name)
        )


def col(name: str, data_type: Optional[int] = None) -> ColumnRef:
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
    return ColumnRef(name, data_type)


FunctionChainArg = Union[ColumnRef, Any]


@dataclass(frozen=True)
class FunctionChainAggregate:
    """One native aggregate, constructed with ``pymilvus.function_chain.agg``."""

    op: str
    input: Optional[ColumnRef] = None

    def __post_init__(self):
        if self.op not in {"count", "sum", "avg", "min", "max", "first", "last", "collect_list"}:
            raise ParamError(message=f"Unknown aggregate: {self.op}")
        if self.input is not None and not isinstance(self.input, ColumnRef):
            raise ParamError(message="aggregate input must be created with col(...)")
        if self.op != "count" and self.input is None:
            raise ParamError(message=f"{self.op} requires an input column")


def _from_param_value(value: schema_pb2.FunctionParamValue) -> Any:
    kind = value.WhichOneof("value")
    if kind == "object_value":
        return {key: _from_param_value(item) for key, item in value.object_value.fields.items()}
    if kind == "array_value":
        return [_from_param_value(item) for item in value.array_value.values]
    if kind is None:
        raise ParamError(message="Function chain parameter value must be set")
    return getattr(value, kind)


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
        if self.expr is not None:
            refs = [arg for arg in self.expr.args if isinstance(arg, ColumnRef)]
            if self.op != "reduce" and any(ref.data_type is not None for ref in refs):
                proto.params["$input_data_types"].CopyFrom(
                    _to_param_value([int(ref.data_type or 0) for ref in refs])
                )
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

    def __init__(
        self,
        stage: FunctionChainStage,
        name: str = "",
        *,
        result_mode: Optional[Union[str, FunctionChainResultMode]] = None,
        input: Optional[Union[str, PostProcessInput]] = None,  # noqa: A002
    ):
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
        self._has_post_options = result_mode is not None or input is not None
        if self._has_post_options and self._stage != FunctionChainStage.POST_PROCESS:
            raise ParamError(message="result_mode and input options require POST_PROCESS")
        self._result_mode = _post_option(
            FunctionChainResultMode.HITS if result_mode is None else result_mode,
            FunctionChainResultMode,
            "result_mode",
        )
        self._post_input = _post_option(
            PostProcessInput.FUSED if input is None else input, PostProcessInput, "input"
        )
        if (
            self._post_input == PostProcessInput.ANN_INPUTS
            and self._result_mode != FunctionChainResultMode.TABLE
        ):
            raise ParamError(message="ANN_INPUTS requires POST TABLE")

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
    def result_mode(self) -> FunctionChainResultMode:
        return self._result_mode

    @property
    def post_process_input(self) -> PostProcessInput:
        return self._post_input

    @property
    def is_table(self) -> bool:
        return (
            self.stage == FunctionChainStage.POST_PROCESS
            and self.result_mode == FunctionChainResultMode.TABLE
        )

    def _require_table(self, operation: str) -> None:
        if not self.is_table:
            raise ParamError(message=f"{operation} requires POST TABLE")

    def _check_source_outputs(self, outputs: Sequence[str], inputs: Sequence[str] = ()):
        if not self.is_table:
            return
        for index, output in enumerate(outputs):
            if output in _SOURCE_COLUMNS and (index >= len(inputs) or inputs[index] != output):
                raise ParamError(message=f"POST source column {output} is read-only")

    @property
    def ops(self) -> Sequence[FunctionChainOp]:
        """Immutable view of operations added to this chain.

        Returns:
            A tuple containing the operations in insertion order.
        """
        return tuple(self._ops)

    def merge(
        self,
        strategy: str,
        *,
        weights: Optional[Sequence[float]] = None,
        norm_score: Optional[bool] = None,
        k: Optional[int] = None,
    ) -> "FunctionChain":
        """Append a Hybrid Search merge operation.

        ``rrf`` accepts ``k`` and optional per-input ``weights``; ``weighted``
        requires ``weights``; ``max``, ``sum``, and ``avg`` optionally normalize
        scores. Strategy-specific
        defaults match the legacy Hybrid Search rankers. Merge semantics are
        validated by the server.

        Example:
            Merge two Hybrid Search result sets with weighted scores::

                chain = FunctionChain(FunctionChainStage.L2_RERANK).merge(
                    "weighted", weights=[0.7, 0.3]
                )

            Use weighted reciprocal rank fusion::

                chain = FunctionChain(FunctionChainStage.L2_RERANK).merge(
                    "rrf", k=60, weights=[0.7, 0.3]
                )
        """
        params: Dict[str, Any] = {"strategy": strategy}
        if strategy == "rrf" and k is None:
            k = 60
        elif strategy in {"weighted", "max", "sum", "avg"} and norm_score is None:
            norm_score = True

        if weights is not None:
            params["weights"] = weights
        if norm_score is not None:
            params["norm_score"] = norm_score
        if k is not None:
            params["k"] = k

        self._ops.append(FunctionChainOp(op="merge", params=params))
        return self

    def map(self, output: Union[str, Sequence[str]], expr: FunctionChainExpr) -> "FunctionChain":
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
        outputs = (output,) if isinstance(output, str) else output
        if (
            not isinstance(outputs, (list, tuple))
            or not outputs
            or not all(isinstance(name, str) and name for name in outputs)
        ):
            raise ParamError(message="Function chain map output must be a non-empty string")
        if len(set(outputs)) != len(outputs):
            raise ParamError(message="Function chain map outputs must be unique")
        if not isinstance(expr, FunctionChainExpr):
            raise ParamError(message="Function chain map expr must be a FunctionChainExpr")
        self._check_source_outputs(outputs)
        self._ops.append(FunctionChainOp(op="map", expr=expr, outputs=tuple(outputs)))
        return self

    def select(
        self,
        inputs: Sequence[Union[str, ColumnRef]],
        outputs: Optional[Sequence[str]] = None,
    ) -> "FunctionChain":
        """Keep selected payload columns, optionally renaming them.

        Struct members use dotted paths, for example ``chunks.text``. Identity
        and row provenance belong to the execution context and survive selection.
        """
        if self.stage not in {
            FunctionChainStage.INGESTION,
            FunctionChainStage.PRE_PROCESS,
            FunctionChainStage.POST_PROCESS,
        }:
            raise ParamError(message="select requires INGESTION, PRE_PROCESS or POST_PROCESS")
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ParamError(message="select inputs must be a non-empty sequence")
        names = tuple(_column_name(value, "select input") for value in inputs)
        aliases = tuple(outputs) if outputs is not None and not isinstance(outputs, str) else names
        if outputs is not None and not isinstance(outputs, (list, tuple)):
            raise ParamError(message="select outputs must be a sequence")
        if len(aliases) != len(names) or not all(
            isinstance(name, str) and name for name in aliases
        ):
            raise ParamError(message="select outputs must match the inputs")
        if len(set(aliases)) != len(aliases):
            raise ParamError(message="select output names must be unique")
        self._check_source_outputs(aliases, names)
        hints = [
            int(value.data_type or 0) if isinstance(value, ColumnRef) else 0 for value in inputs
        ]
        params = {"$input_data_types": hints} if any(hints) else {}
        self._ops.append(FunctionChainOp(op="select", inputs=names, outputs=aliases, params=params))
        return self

    def explode(
        self,
        source: Union[str, ColumnRef],
        output: Optional[str] = None,
        *,
        outer: bool = False,
        position: Optional[str] = None,
    ) -> "FunctionChain":
        """Expand one List column, repeating its parent's other columns."""
        source = _column_name(source, "explode input")
        target = _column_name(output if output is not None else source, "explode output")
        params = self._expansion_params(outer, position)
        if position == target:
            raise ParamError(message="explode position must differ from the output")
        outputs = (target, position) if position is not None else (target,)
        self._check_source_outputs(outputs)
        self._ops.append(
            FunctionChainOp(op="explode", inputs=(source,), outputs=outputs, params=params)
        )
        return self

    def flat_map(
        self,
        output: str,
        expr: FunctionChainExpr,
        *,
        outer: bool = False,
        position: Optional[str] = None,
    ) -> "FunctionChain":
        """Evaluate a List-valued expression and emit one row per element."""
        target = _column_name(output, "flat_map output")
        if not isinstance(expr, FunctionChainExpr):
            raise ParamError(message="flat_map expr must be a FunctionChainExpr")
        params = self._expansion_params(outer, position)
        if position == target:
            raise ParamError(message="flat_map position must differ from the output")
        outputs = (target, position) if position is not None else (target,)
        self._check_source_outputs(outputs)
        self._ops.append(FunctionChainOp(op="flat_map", expr=expr, outputs=outputs, params=params))
        return self

    def filter(self, expr: FunctionChainExpr) -> "FunctionChain":
        """Keep INGESTION or POST TABLE rows selected by a Boolean expression."""
        if self.stage != FunctionChainStage.INGESTION and not self.is_table:
            raise ParamError(message="filter requires INGESTION or POST TABLE")
        if not isinstance(expr, FunctionChainExpr):
            raise ParamError(message="filter expr must be a FunctionChainExpr")
        self._ops.append(FunctionChainOp(op="filter", expr=expr))
        return self

    def _expansion_params(self, outer: bool, position: Optional[str]) -> Dict[str, Any]:
        if self.stage != FunctionChainStage.INGESTION and not self.is_table:
            raise ParamError(message="row expansion requires INGESTION or POST TABLE")
        if not isinstance(outer, bool):
            raise ParamError(message="outer must be a bool")
        params = {"outer": outer}
        if position is not None:
            _column_name(position, "position")
        return params

    def concat(self) -> "FunctionChain":
        """Combine logical groups in stable order, retaining duplicate rows."""
        self._require_table("concat")
        self._ops.append(FunctionChainOp(op="concat"))
        return self

    def group_by(
        self, *keys: Union[str, ColumnRef], aliases: Optional[Sequence[str]] = None
    ) -> "GroupedFunctionChain":
        """Declare keys for the next aggregate/reduce, without emitting a rerank op."""
        self._require_table("group_by")
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = tuple(keys[0])
        if not keys:
            raise ParamError(message="group_by requires at least one key")
        refs = tuple(key if isinstance(key, ColumnRef) else ColumnRef(key) for key in keys)
        names = tuple(ref.name for ref in refs)
        if aliases is not None and not isinstance(aliases, (list, tuple)):
            raise ParamError(message="group_by aliases must be a sequence")
        outputs = names if aliases is None else tuple(aliases)
        if len(outputs) != len(names) or not all(
            isinstance(name, str) and name for name in outputs
        ):
            raise ParamError(message="group_by aliases must match the keys")
        if len(set(outputs)) != len(outputs):
            raise ParamError(message="group_by aliases must be unique")
        self._check_source_outputs(outputs, names)
        return GroupedFunctionChain(self, refs, outputs)

    def aggregate(self, aggregates: Mapping[str, FunctionChainAggregate]) -> "FunctionChain":
        """Aggregate each current logical group; call concat first to combine NQ."""
        return self._aggregate((), (), aggregates)

    def _aggregate(
        self,
        keys: Sequence[ColumnRef],
        key_outputs: Sequence[str],
        aggregates: Mapping[str, FunctionChainAggregate],
    ) -> "FunctionChain":
        self._require_table("aggregate")
        if not isinstance(aggregates, Mapping) or not aggregates:
            raise ParamError(message="aggregate requires a non-empty alias-to-aggregate mapping")
        outputs = list(key_outputs)
        inputs = list(keys)
        specs = []
        for alias, spec in aggregates.items():
            if not isinstance(alias, str) or not alias:
                raise ParamError(message="aggregate aliases must be non-empty strings")
            if alias.startswith("$"):
                raise ParamError(message="aggregate aliases must be ordinary column names")
            if not isinstance(spec, FunctionChainAggregate):
                raise ParamError(message="aggregate values must be built with agg helpers")
            outputs.append(alias)
            value = {"op": spec.op}
            if spec.input is not None:
                value["input"] = spec.input.name
                inputs.append(spec.input)
            specs.append(value)
        if len(set(outputs)) != len(outputs):
            raise ParamError(message="aggregate output names must be unique")
        self._check_source_outputs(outputs, [key.name for key in keys])
        params = {"keys": [key.name for key in keys], "aggregates": specs}
        hints = [int(ref.data_type or 0) for ref in inputs]
        if any(hints):
            params["$input_data_types"] = hints
        self._ops.append(
            FunctionChainOp(
                op="aggregate",
                inputs=tuple(ref.name for ref in inputs),
                outputs=tuple(outputs),
                params=params,
            )
        )
        return self

    def reduce(
        self, outputs: Union[str, Sequence[str]], expr: FunctionChainExpr
    ) -> "FunctionChain":
        """Run a reducing Python UDF once per current logical group."""
        return self._reduce((), (), outputs, expr)

    def _reduce(
        self,
        keys: Sequence[ColumnRef],
        key_outputs: Sequence[str],
        outputs: Union[str, Sequence[str]],
        expr: FunctionChainExpr,
    ) -> "FunctionChain":
        self._require_table("reduce")
        if not isinstance(expr, FunctionChainExpr) or expr.name != "py_udf":
            raise ParamError(message="reduce requires fn.py_udf")
        outputs = (outputs,) if isinstance(outputs, str) else outputs
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise ParamError(message="reduce outputs must be a non-empty sequence")
        for output in outputs:
            if not isinstance(output, str) or not output:
                raise ParamError(message="reduce aliases must be non-empty strings")
            if output.startswith("$"):
                raise ParamError(message="reduce aliases must be ordinary column names")
        aliases = tuple(key_outputs) + tuple(outputs)
        if len(set(aliases)) != len(aliases):
            raise ParamError(message="reduce output names must be unique")
        self._check_source_outputs(aliases, [key.name for key in keys])
        params = {"keys": [key.name for key in keys]}
        # Reduce inputs are composed by the server from key occurrences followed
        # by expression arguments. Preserve hints in exactly that order.
        refs = list(keys) + [arg for arg in expr.args if isinstance(arg, ColumnRef)]
        hints = [int(ref.data_type or 0) for ref in refs]
        if any(hints):
            params["$input_data_types"] = hints
        self._ops.append(FunctionChainOp(op="reduce", expr=expr, outputs=aliases, params=params))
        return self

    @classmethod
    def from_proto(cls, proto: schema_pb2.FunctionChain) -> "FunctionChain":
        """Restore a collection's persisted chain, including typed parameters."""
        options = {}
        if proto.HasField("post_process_options"):
            options = {
                "result_mode": proto.post_process_options.result_mode,
                "input": proto.post_process_options.input,
            }
        chain = cls(proto.stage, proto.name, **options)
        for operation in proto.ops:
            expression = None
            if operation.HasField("expr"):
                expression = FunctionChainExpr(
                    operation.expr.name,
                    args=tuple(
                        (
                            ColumnRef(arg.column.name)
                            if arg.WhichOneof("arg") == "column"
                            else _from_param_value(arg.literal)
                        )
                        for arg in operation.expr.args
                    ),
                    params={
                        key: _from_param_value(value)
                        for key, value in operation.expr.params.items()
                    },
                )
            chain._ops.append(
                FunctionChainOp(
                    operation.op,
                    expression,
                    tuple(operation.inputs),
                    tuple(operation.outputs),
                    {key: _from_param_value(value) for key, value in operation.params.items()},
                )
            )
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return MessageToDict(self.to_proto(), preserving_proto_field_name=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FunctionChain":
        return cls.from_proto(ParseDict(dict(value), schema_pb2.FunctionChain()))

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
        if self._has_post_options:
            proto.post_process_options.CopyFrom(
                schema_pb2.PostProcessOptions(
                    result_mode=int(self._result_mode), input=int(self._post_input)
                )
            )
        proto.ops.extend([op.to_proto() for op in self._ops])
        return proto


class GroupedFunctionChain:
    """Keys pending an aggregate/reduce; no standalone group_by op is emitted."""

    def __init__(self, chain: FunctionChain, keys: Sequence[ColumnRef], outputs: Sequence[str]):
        self._chain = chain
        self._keys = keys
        self._outputs = outputs

    def aggregate(self, aggregates: Mapping[str, FunctionChainAggregate]) -> FunctionChain:
        return self._chain._aggregate(self._keys, self._outputs, aggregates)

    def reduce(self, outputs: Union[str, Sequence[str]], expr: FunctionChainExpr) -> FunctionChain:
        return self._chain._reduce(self._keys, self._outputs, outputs, expr)
