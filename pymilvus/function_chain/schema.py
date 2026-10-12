"""Generated-field information derived from a persisted ingestion plan."""

from typing import List, Set

from .chain import FunctionChain


def generated_fields(chains: List[FunctionChain]) -> Set[str]:
    produced, required = set(), set()
    for chain in chains:
        for op in chain.to_proto().ops:
            inputs = list(op.inputs)
            if op.HasField("expr"):
                inputs.extend(arg.column.name for arg in op.expr.args if arg.HasField("column"))
            for name in inputs:
                root = name.split(".", 1)[0]
                if name not in produced and root not in produced:
                    required.add(root)
            if op.op == "select":
                produced = {
                    alias
                    for source, alias in zip(op.inputs, op.outputs or op.inputs)
                    if source in produced or source.split(".", 1)[0] in produced or alias != source
                }
            else:
                produced.update(op.outputs)
    return produced - required
