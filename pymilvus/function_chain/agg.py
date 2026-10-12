"""Native aggregates for POST TABLE function chains."""

from typing import Optional

from .chain import ColumnRef, FunctionChainAggregate


def count(column: Optional[ColumnRef] = None) -> FunctionChainAggregate:
    """Count every row, or non-null values when a column is supplied."""
    return FunctionChainAggregate("count", column)


def sum(column: ColumnRef) -> FunctionChainAggregate:  # noqa: A001
    return FunctionChainAggregate("sum", column)


def avg(column: ColumnRef) -> FunctionChainAggregate:
    return FunctionChainAggregate("avg", column)


def min(column: ColumnRef) -> FunctionChainAggregate:  # noqa: A001
    return FunctionChainAggregate("min", column)


def max(column: ColumnRef) -> FunctionChainAggregate:  # noqa: A001
    return FunctionChainAggregate("max", column)


def first(column: ColumnRef) -> FunctionChainAggregate:
    """Take the first value, including null."""
    return FunctionChainAggregate("first", column)


def last(column: ColumnRef) -> FunctionChainAggregate:
    """Take the last value, including null."""
    return FunctionChainAggregate("last", column)


def collect_list(column: ColumnRef) -> FunctionChainAggregate:
    """Collect values in input order, retaining nulls; an empty group yields []."""
    return FunctionChainAggregate("collect_list", column)
