import json
from pathlib import Path

import pytest
from pymilvus import build_bloom_filter
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError


def test_build_bloom_filter_matches_parquet_sbbf_vector():
    members = [-(1 << 63), -1, 0, 1, 2, 42, 1000000007, (1 << 63) - 1]
    expected = bytes.fromhex(
        "4d424631010001000800000000000000fca9f1d24d62503f0100000000000000"
        "0117100a804845062834804200440612100448a40680089c4040542838410046"
    )

    assert build_bloom_filter(members, fpr=0.001) == expected


def test_build_bloom_filter_matches_parquet_sbbf_string_vector():
    members = ["", "a", "milvus", "bloom", "日本語", "🚀🚀", "hello world"]
    expected = bytes.fromhex(
        "4d424631010001000700000000000000fca9f1d24d62503f0100000000000000"
        "10002c24010e020492110042181000b08a40860000284209010114a8015410c0"
    )

    assert build_bloom_filter(members, fpr=0.001) == expected


def test_build_bloom_filter_matches_arrow_cpp_fixture():
    fixture_path = Path(__file__).parent / "testdata" / "cpp_generated_100_int64.json"
    fixture = json.loads(fixture_path.read_text())

    assert fixture["generator"] == "apache_arrow_parquet_block_split_bloom_filter"
    assert len(fixture["int_values"]) == 100
    assert build_bloom_filter(
        [int(value) for value in fixture["int_values"]], fixture["fpr"]
    ) == bytes.fromhex(fixture["blob_hex"])


def test_build_bloom_filter_and_template_bytes_value():
    blob = build_bloom_filter(["alice", "bob", "小明"], fpr=0.01)

    values = Prepare.prepare_expression_template({"bf": blob})

    assert values["bf"].bytes_val == blob


@pytest.mark.parametrize(
    "members,fpr",
    [([1, "mixed"], 0.001), ([True], 0.001), ([1], 0.5)],
)
def test_build_bloom_filter_rejects_invalid_input(members, fpr):
    with pytest.raises(ParamError):
        build_bloom_filter(members, fpr=fpr)
