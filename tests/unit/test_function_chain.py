import pytest
from pymilvus import FunctionChain, FunctionChainStage, FunctionType
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.function_chain import FunctionChainExpr, FunctionChainOp, col, fn
from pymilvus.grpc_gen import schema_pb2
from pymilvus.orm.schema import Function


def _prepare_search(**kwargs):
    base = {
        "collection_name": "c",
        "anns_field": "emb",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 10,
        "data": [[0.1] * 4],
    }
    base.update(kwargs)
    return Prepare.search_requests_with_expr(**base)


class TestColumnRef:
    def test_col(self):
        ref = col("$score")
        proto = ref.to_proto()
        assert proto.WhichOneof("arg") == "column"
        assert proto.column.name == "$score"

    @pytest.mark.parametrize("bad", ["", None, 1])
    def test_bad_col(self, bad):
        with pytest.raises(ParamError):
            col(bad)


class TestFnHelpers:
    def test_num_combine_sum(self):
        expr = fn.num_combine(col("$score"), col("ts"), mode="sum")
        proto = expr.to_proto()
        assert proto.name == "num_combine"
        assert [arg.column.name for arg in proto.args] == ["$score", "ts"]
        assert proto.params["mode"].string_value == "sum"

    def test_num_combine_weighted(self):
        expr = fn.num_combine(col("a"), col("b"), mode="weighted", weights=[0.7, 0.3])
        weights = expr.to_proto().params["weights"].array_value.values
        assert [value.double_value for value in weights] == [0.7, 0.3]

    @pytest.mark.parametrize(
        "call",
        [
            lambda: fn.num_combine(col("a")),
            lambda: fn.num_combine(col("a"), col("b"), mode="unknown"),
            lambda: fn.num_combine(col("a"), col("b"), mode="weighted"),
            lambda: fn.num_combine(col("a"), col("b"), mode="sum", weights=[1.0, 2.0]),
        ],
    )
    def test_bad_num_combine(self, call):
        with pytest.raises(ParamError):
            call()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: fn.num_combine(col("a"), "b"),
            lambda: fn.num_combine(col("a"), col("b"), mode="weighted", weights=[1.0]),
            lambda: fn.num_combine(col("a"), col("b"), mode="weighted", weights=[1.0, True]),
        ],
    )
    def test_bad_num_combine_inputs(self, call):
        with pytest.raises(ParamError):
            call()

    def test_decay(self):
        expr = fn.decay(col("ts"), function="linear", origin=100, scale=10, offset=1, decay=0.5)
        proto = expr.to_proto()
        assert proto.name == "decay"
        assert proto.args[0].column.name == "ts"
        assert proto.params["function"].string_value == "linear"
        assert proto.params["origin"].int64_value == 100
        assert proto.params["scale"].int64_value == 10
        assert proto.params["offset"].int64_value == 1
        assert proto.params["decay"].double_value == 0.5

    @pytest.mark.parametrize(
        "call",
        [
            lambda: fn.decay("ts", function="linear", origin=100, scale=10),
            lambda: fn.decay(col("ts"), function="unknown", origin=100, scale=10),
            lambda: fn.decay(col("ts"), function="linear", origin=True, scale=10),
            lambda: fn.decay(col("ts"), function="linear", origin=100, scale="10"),
        ],
    )
    def test_bad_decay(self, call):
        with pytest.raises(ParamError):
            call()

    def test_round_decimal(self):
        expr = fn.round_decimal(col("$score"), decimal=3)
        proto = expr.to_proto()
        assert proto.name == "round_decimal"
        assert proto.params["decimal"].int64_value == 3

    @pytest.mark.parametrize("decimal", [-1, 7, True, 1.1])
    def test_bad_round_decimal(self, decimal):
        with pytest.raises(ParamError):
            fn.round_decimal(col("$score"), decimal=decimal)

    def test_bad_round_decimal_column(self):
        with pytest.raises(ParamError):
            fn.round_decimal("$score", decimal=3)

    def test_rerank_model(self):
        expr = fn.rerank_model(
            col("doc"),
            queries=["hello"],
            provider="voyageai",
            truncation=True,
            nested={"k": [1, "v"]},
        )
        proto = expr.to_proto()
        assert proto.name == "rerank_model"
        assert proto.params["queries"].array_value.values[0].string_value == "hello"
        assert proto.params["provider"].string_value == "voyageai"
        assert proto.params["truncation"].bool_value is True
        assert (
            proto.params["nested"].object_value.fields["k"].array_value.values[0].int64_value == 1
        )

    @pytest.mark.parametrize(
        "call",
        [
            lambda: fn.rerank_model("doc", queries=["hello"]),
            lambda: fn.rerank_model(col("doc"), queries=[]),
            lambda: fn.rerank_model(col("doc"), queries=[""]),
        ],
    )
    def test_bad_rerank_model(self, call):
        with pytest.raises(ParamError):
            call()


class TestFunctionChainExpr:
    def test_literal_args_and_param_values(self):
        expr = FunctionChainExpr(
            "literal_test",
            args=(True, 7, 1.5, "value", b"bytes", [1, "two"], {"nested": False}),
            params={"ok": True},
        )

        proto = expr.to_proto()

        assert proto.args[0].literal.bool_value is True
        assert proto.args[1].literal.int64_value == 7
        assert proto.args[2].literal.double_value == 1.5
        assert proto.args[3].literal.string_value == "value"
        assert proto.args[4].literal.bytes_value == b"bytes"
        assert proto.args[5].literal.array_value.values[1].string_value == "two"
        assert proto.args[6].literal.object_value.fields["nested"].bool_value is False
        assert proto.params["ok"].bool_value is True

    @pytest.mark.parametrize(
        "call",
        [
            lambda: FunctionChainExpr("", params={}),
            lambda: FunctionChainExpr("bad", params=[]),
            lambda: FunctionChainExpr("bad", args=(None,)).to_proto(),
            lambda: FunctionChainExpr("bad", args=(object(),)).to_proto(),
            lambda: FunctionChainExpr("bad", args=({"nested": {1: "bad"}},)).to_proto(),
            lambda: FunctionChainExpr("bad", params={"": 1}).to_proto(),
        ],
    )
    def test_bad_expr(self, call):
        with pytest.raises(ParamError):
            call()


class TestFunctionChainOp:
    def test_op_without_expr(self):
        proto = FunctionChainOp(op="noop", inputs=("score",), params={"enabled": True}).to_proto()

        assert proto.op == "noop"
        assert proto.inputs == ["score"]
        assert proto.params["enabled"].bool_value is True
        assert not proto.HasField("expr")

    @pytest.mark.parametrize(
        "call",
        [
            lambda: FunctionChainOp(op=""),
            lambda: FunctionChainOp(op="map", expr="bad"),
            lambda: FunctionChainOp(op="map", params=[]),
        ],
    )
    def test_bad_op(self, call):
        with pytest.raises(ParamError):
            call()


class TestFunctionChain:
    def test_properties(self):
        chain = FunctionChain(FunctionChainStage.L2_RERANK, name="props")

        assert chain.name == "props"
        assert chain.stage == FunctionChainStage.L2_RERANK
        assert chain.ops == ()

    def test_map_sort_limit_to_proto(self):
        chain = (
            FunctionChain(FunctionChainStage.L2_RERANK, name="c1")
            .map("freshness", fn.decay(col("ts"), function="linear", origin=100, scale=10))
            .map("$score", fn.num_combine(col("$score"), col("freshness"), mode="sum"))
            .sort(col("$score"), desc=True, tie_break_col=col("$id"))
            .limit(20, offset=2)
        )

        proto = chain.to_proto()
        assert proto.name == "c1"
        assert proto.stage == schema_pb2.FunctionChainStageL2Rerank
        assert [op.op for op in proto.ops] == ["map", "map", "sort", "limit"]
        assert proto.ops[0].outputs == ["freshness"]
        assert proto.ops[0].expr.name == "decay"
        assert proto.ops[2].inputs == ["$score", "$id"]
        assert proto.ops[2].params["column"].string_value == "$score"
        assert proto.ops[2].params["desc"].bool_value is True
        assert proto.ops[2].params["tie_break_col"].string_value == "$id"
        assert proto.ops[3].params["limit"].int64_value == 20
        assert proto.ops[3].params["offset"].int64_value == 2

    def test_sort_with_string_columns(self):
        proto = (
            FunctionChain(FunctionChainStage.L2_RERANK).sort("score", tie_break_col="id").to_proto()
        )

        assert proto.ops[0].inputs == ["score", "id"]

    @pytest.mark.parametrize(
        "call",
        [
            lambda: FunctionChain(999),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK, name=1),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).map(
                "", fn.round_decimal(col("$score"), decimal=3)
            ),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).map("score", "bad"),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).sort("score", desc="yes"),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).sort(""),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).sort("score", tie_break_col=""),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).limit(0),
            lambda: FunctionChain(FunctionChainStage.L2_RERANK).limit(10, offset=-1),
        ],
    )
    def test_bad_chain(self, call):
        with pytest.raises(ParamError):
            call()

    def test_bad_stage_for_search(self):
        chain = FunctionChain(FunctionChainStage.L1_RERANK)
        with pytest.raises(ParamError, match="L2_RERANK"):
            _prepare_search(function_chains=[chain])


class TestSearchIntegration:
    def test_prepare_search_with_function_chains(self):
        chain = FunctionChain(FunctionChainStage.L2_RERANK).map(
            "$score", fn.num_combine(col("$score"), col("ts"), mode="sum")
        )
        request = _prepare_search(function_chains=chain)
        assert len(request.function_chains) == 1
        assert request.function_chains[0].stage == schema_pb2.FunctionChainStageL2Rerank
        assert request.function_chains[0].ops[0].expr.name == "num_combine"

    def test_prepare_search_rejects_ranker_and_function_chains(self):
        chain = FunctionChain(FunctionChainStage.L2_RERANK)
        ranker = Function(
            name="rerank",
            function_type=FunctionType.RERANK,
            input_field_names=["text"],
            params={"provider": "mock"},
        )
        with pytest.raises(ParamError, match="function_chains and ranker"):
            _prepare_search(function_chains=[chain], ranker=ranker)

    def test_prepare_search_allows_empty_function_chains_with_ranker(self):
        ranker = Function(
            name="rerank",
            function_type=FunctionType.RERANK,
            input_field_names=["text"],
            params={"provider": "mock"},
        )

        request = _prepare_search(function_chains=[], ranker=ranker)

        assert len(request.function_chains) == 0
        assert len(request.function_score.functions) == 1
        assert request.function_score.functions[0].name == "rerank"

    @pytest.mark.parametrize("bad", [1, [object()], {"x": 1}])
    def test_prepare_search_rejects_bad_function_chains(self, bad):
        with pytest.raises(ParamError, match="function_chains"):
            _prepare_search(function_chains=bad)

    def test_hybrid_search_rejects_function_chains(self):
        with pytest.raises(ParamError, match="hybrid_search"):
            GrpcHandler.hybrid_search(
                object.__new__(GrpcHandler),
                collection_name="c",
                reqs=[],
                rerank=None,
                limit=10,
                function_chains=[FunctionChain(FunctionChainStage.L2_RERANK)],
            )
