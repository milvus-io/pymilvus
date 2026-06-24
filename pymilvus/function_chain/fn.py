from typing import Any, List, Optional, Union

from pymilvus.exceptions import ParamError

from .chain import ColumnRef, FunctionChainExpr

_NUM_COMBINE_MODES = {"multiply", "sum", "max", "min", "avg", "weighted"}
_DECAY_FUNCTIONS = {"gauss", "exp", "linear"}


def _check_column_ref(value: Any, param_name: str) -> None:
    """Validate that a function argument is a column reference.

    Use this helper before building a function-chain expression that requires an
    input column. Callers should pass values created by ``col("field_name")``;
    raw strings are rejected so public helper functions can provide consistent
    error messages.

    Args:
        value: The value to validate.
        param_name: The argument name to include in the error message.

    Raises:
        ParamError: If ``value`` is not a ``ColumnRef`` created by ``col(...)``.
    """
    if not isinstance(value, ColumnRef):
        raise ParamError(message=f"{param_name} must be created by col(...)")


def num_combine(
    *cols: ColumnRef,
    mode: str = "sum",
    weights: Optional[List[float]] = None,
) -> FunctionChainExpr:
    """Build a numeric combination expression for two or more columns.

    Use this helper in ``FunctionChain.map`` when a function chain needs to
    combine multiple numeric values, such as the current score and a freshness
    score, into one output column.

    Supported ``mode`` values are:
        - ``"sum"``: Add all input columns.
        - ``"multiply"``: Multiply all input columns.
        - ``"max"``: Use the maximum input value.
        - ``"min"``: Use the minimum input value.
        - ``"avg"``: Use the average input value.
        - ``"weighted"``: Compute a weighted sum. ``weights`` is required and
          must contain one numeric value for each input column.

    Example:
        >>> from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn
        >>> chain = FunctionChain(FunctionChainStage.L2_RERANK).map(
        ...     "$score",
        ...     fn.num_combine(col("$score"), col("freshness"), mode="sum"),
        ... )
        >>> weighted = fn.num_combine(
        ...     col("semantic_score"),
        ...     col("keyword_score"),
        ...     mode="weighted",
        ...     weights=[0.7, 0.3],
        ... )

    Args:
        *cols: Two or more input columns created by ``col(...)``.
        mode: Combination strategy. Defaults to ``"sum"``.
        weights: Per-column weights. Only valid when ``mode="weighted"``.

    Returns:
        A ``FunctionChainExpr`` that can be passed to ``FunctionChain.map``.

    Raises:
        ParamError: If fewer than two columns are provided, an argument is not a
            ``ColumnRef``, ``mode`` is unsupported, or ``weights`` does not match
            the requirements for weighted mode.
    """
    if len(cols) < 2:
        raise ParamError(message="fn.num_combine requires at least two columns")
    for value in cols:
        _check_column_ref(value, "fn.num_combine argument")
    if mode not in _NUM_COMBINE_MODES:
        raise ParamError(
            message=f"Unsupported num_combine mode: {mode}, expected one of {sorted(_NUM_COMBINE_MODES)}"
        )

    params = {"mode": mode}
    if weights is not None:
        if mode != "weighted":
            raise ParamError(message="num_combine weights can only be used with mode='weighted'")
        if not isinstance(weights, list) or len(weights) != len(cols):
            raise ParamError(message="num_combine weights must be a list matching the column count")
        for weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ParamError(message="num_combine weights must be numeric")
        params["weights"] = [float(weight) for weight in weights]
    elif mode == "weighted":
        raise ParamError(message="num_combine mode='weighted' requires weights")

    return FunctionChainExpr("num_combine", args=cols, params=params)


def decay(
    value: ColumnRef,
    *,
    function: str,
    origin: Union[int, float],
    scale: Union[int, float],
    offset: Union[int, float] = 0,
    decay: float = 0.5,
) -> FunctionChainExpr:
    """Build a decay scoring expression for a numeric column.

    Use this helper in ``FunctionChain.map`` to turn a numeric value, such as a
    timestamp, distance, or popularity signal, into a score that decreases as
    the value moves away from ``origin``. The resulting expression can be mapped
    to an intermediate column and then combined with other scores.

    Supported decay functions are ``"gauss"``, ``"exp"``, and ``"linear"``.
    ``scale`` controls how quickly the score decays, ``offset`` defines a range
    around ``origin`` with no decay, and ``decay`` is the target score reached at
    ``scale``.

    Example:
        >>> from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn
        >>> chain = (
        ...     FunctionChain(FunctionChainStage.L2_RERANK)
        ...     .map(
        ...         "freshness",
        ...         fn.decay(col("timestamp"), function="linear", origin=1700000000, scale=86400),
        ...     )
        ...     .map("$score", fn.num_combine(col("$score"), col("freshness"), mode="sum"))
        ... )

    Args:
        value: Input numeric column created by ``col(...)``.
        function: Decay function name: ``"gauss"``, ``"exp"``, or ``"linear"``.
        origin: Best or reference value where the decay score starts.
        scale: Distance from ``origin`` where the score reaches ``decay``.
        offset: Non-decaying distance around ``origin``. Defaults to ``0``.
        decay: Score value at ``scale``. Defaults to ``0.5``.

    Returns:
        A ``FunctionChainExpr`` that can be passed to ``FunctionChain.map``.

    Raises:
        ParamError: If ``value`` is not a ``ColumnRef``, ``function`` is
            unsupported, or any numeric parameter is not an int or float.
    """
    _check_column_ref(value, "fn.decay value")
    if function not in _DECAY_FUNCTIONS:
        raise ParamError(
            message=f"Unsupported decay function: {function}, expected one of {sorted(_DECAY_FUNCTIONS)}"
        )
    for name, param in {
        "origin": origin,
        "scale": scale,
        "offset": offset,
        "decay": decay,
    }.items():
        if isinstance(param, bool) or not isinstance(param, (int, float)):
            raise ParamError(message=f"decay {name} must be numeric")

    return FunctionChainExpr(
        "decay",
        args=(value,),
        params={
            "function": function,
            "origin": origin,
            "scale": scale,
            "offset": offset,
            "decay": decay,
        },
    )


def round_decimal(value: ColumnRef, *, decimal: int) -> FunctionChainExpr:
    """Build an expression that rounds a numeric column to fixed decimals.

    Use this helper in ``FunctionChain.map`` when a function chain should round
    a floating-point score or feature before sorting, combining, or returning it.
    The number of decimal places must be between ``0`` and ``6``.

    Example:
        >>> from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn
        >>> chain = FunctionChain(FunctionChainStage.L2_RERANK).map(
        ...     "rounded_score",
        ...     fn.round_decimal(col("$score"), decimal=3),
        ... )

    Args:
        value: Input numeric column created by ``col(...)``.
        decimal: Number of decimal places to keep. Must be an integer in
            ``[0, 6]``.

    Returns:
        A ``FunctionChainExpr`` that can be passed to ``FunctionChain.map``.

    Raises:
        ParamError: If ``value`` is not a ``ColumnRef`` or ``decimal`` is not an
            integer in ``[0, 6]``.
    """
    _check_column_ref(value, "fn.round_decimal value")
    if isinstance(decimal, bool) or not isinstance(decimal, int) or decimal < 0 or decimal > 6:
        raise ParamError(message="round_decimal decimal must be an integer in [0, 6]")
    return FunctionChainExpr("round_decimal", args=(value,), params={"decimal": decimal})


def rerank_model(
    value: ColumnRef,
    *,
    queries: List[str],
    **provider_params,
) -> FunctionChainExpr:
    """Build an expression that reranks a column with an external rerank model.

    Use this helper in ``FunctionChain.map`` when a rerank stage should call a
    model provider using one input column, such as a document text field. Pass
    the search queries through ``queries`` and pass provider-specific options as
    keyword arguments. Provider options are forwarded as expression parameters.

    Example:
        >>> from pymilvus.function_chain import FunctionChain, FunctionChainStage, col, fn
        >>> chain = FunctionChain(FunctionChainStage.L2_RERANK).map(
        ...     "$score",
        ...     fn.rerank_model(
        ...         col("text"),
        ...         queries=["how to use Milvus function chains"],
        ...         provider="voyageai",
        ...         model="rerank-lite-1",
        ...         truncation=True,
        ...     ),
        ... )

    Args:
        value: Input column created by ``col(...)``. This is typically a text
            field used by the rerank provider.
        queries: Non-empty list of query strings to score against ``value``.
        **provider_params: Provider-specific parameters, such as ``provider``,
            ``model``, ``top_k``, or other options supported by the service.

    Returns:
        A ``FunctionChainExpr`` that can be passed to ``FunctionChain.map``.

    Raises:
        ParamError: If ``value`` is not a ``ColumnRef`` or ``queries`` is not a
            non-empty list of non-empty strings.
    """
    _check_column_ref(value, "fn.rerank_model value")
    if (
        not isinstance(queries, list)
        or not queries
        or not all(isinstance(query, str) and query for query in queries)
    ):
        raise ParamError(message="rerank_model queries must be a non-empty list of strings")
    params = {"queries": queries}
    params.update(provider_params)
    return FunctionChainExpr("rerank_model", args=(value,), params=params)
