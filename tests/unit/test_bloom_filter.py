import json
import random
import struct
from pathlib import Path

import pytest
from pymilvus import build_bloom_filter
from pymilvus.client import bloom_filter
from pymilvus.client.bloom_filter import _xxh64_int64_python, _xxh64_python
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError

try:
    import xxhash
except ImportError:
    xxhash = None


def test_build_bloom_filter_matches_parquet_sbbf_vector():
    members = [-(1 << 63), -1, 0, 1, 2, 42, 1000000007, (1 << 63) - 1]
    expected = bytes.fromhex(
        "4d424631010001000800000000000000fca9f1d24d62503f0100000001000000"
        "0117100a804845062834804200440612100448a40680089c4040542838410046"
    )

    assert build_bloom_filter(members, fpr=0.001) == expected


def test_build_bloom_filter_matches_parquet_sbbf_string_vector():
    members = ["", "a", "milvus", "bloom", "日本語", "🚀🚀", "hello world"]
    expected = bytes.fromhex(
        "4d424631010001000700000000000000fca9f1d24d62503f0100000002000000"
        "10002c24010e020492110042181000b08a40860000284209010114a8015410c0"
    )

    assert build_bloom_filter(members, fpr=0.001) == expected


@pytest.mark.parametrize(
    "length,expected_hex",
    [
        (
            31,
            "4d424631010001000100000000000000fca9f1d24d62503f0100000002000000"
            "0000000200100000000080000004000000000200000008002000000000000200",
        ),
        (
            32,
            "4d424631010001000100000000000000fca9f1d24d62503f0100000002000000"
            "0000800020000000020000000000020002000000000080000000800000000080",
        ),
        (
            33,
            "4d424631010001000100000000000000fca9f1d24d62503f0100000002000000"
            "0000800000000001000000040001000020000000000020000800000040000000",
        ),
    ],
)
def test_build_bloom_filter_matches_go_xxh64_boundary_vectors(length, expected_hex):
    assert build_bloom_filter(["a" * length], fpr=0.001) == bytes.fromhex(expected_hex)


def test_xxh64_int64_python_matches_generic():
    """The INT64 fast path must stay bit-identical to the generic XXH64 it specialises."""
    rng = random.Random(20260728)
    values = [0, 1, -1, 2, -2, 42, -(1 << 63), (1 << 63) - 1, (1 << 62), -(1 << 62)]
    values += [rng.randrange(-(1 << 63), 1 << 63) for _ in range(2000)]

    for value in values:
        assert _xxh64_int64_python(value) == _xxh64_python(struct.pack("<q", value)), value


@pytest.mark.parametrize(
    "members",
    [
        [-(1 << 63), -1, 0, 1, 42, (1 << 63) - 1],
        ["", "a", "milvus", "日本語", "🚀🚀", "x" * 40],
    ],
)
def test_build_bloom_filter_fallback_matches_default_path(monkeypatch, members):
    """Forcing the pure-Python hashes must reproduce the blob the default path builds.

    Keeps the fallback exercised even when the optional C accelerator is installed.
    """
    expected = build_bloom_filter(members, fpr=0.001)

    monkeypatch.setattr(bloom_filter, "_xxh64", bloom_filter._xxh64_python)
    monkeypatch.setattr(bloom_filter, "_xxh64_int64", bloom_filter._xxh64_int64_python)

    assert build_bloom_filter(members, fpr=0.001) == expected


@pytest.mark.parametrize("count", [1, 2, 100, (1 << 14) + 3])
def test_int64_vectorised_matches_scalar(monkeypatch, count):
    """The numpy INT64 path must agree with the scalar loop, including across chunk boundaries."""
    rng = random.Random(20260728)
    members = [-(1 << 63), (1 << 63) - 1, 0, -1]
    members += [rng.randrange(-(1 << 63), 1 << 63) for _ in range(count)]

    vectorised = build_bloom_filter(members, fpr=0.001)

    monkeypatch.setattr(bloom_filter, "_fill_int64_vectorised", bloom_filter._fill_scalar_int64)
    assert build_bloom_filter(members, fpr=0.001) == vectorised


@pytest.mark.parametrize(
    "members",
    [[1, True], [True, 1], [1, "x"], [1, None], [1, 1 << 63], [1, -(1 << 63) - 1], [1, 1.5]],
)
def test_int64_vectorised_rejects_bad_members(members):
    """Type and range errors raised inside numpy must surface as ParamError, not leak out."""
    with pytest.raises(ParamError):
        build_bloom_filter(members, fpr=0.001)


@pytest.mark.skipif(xxhash is None, reason="requires the optional bloom_filter extra")
def test_xxh64_python_matches_xxhash_c():
    """The optional C accelerator must agree with the fallback on every input shape."""
    rng = random.Random(20260728)
    payloads = [b"", b"a", b"milvus", "日本語".encode(), "🚀🚀".encode()]
    # Straddle the 32-byte stripe boundary and both tail loops (8-byte lane, 4-byte, 1-byte).
    payloads += [b"a" * n for n in range(80)]
    payloads += [rng.randbytes(rng.randrange(0, 200)) for _ in range(500)]

    for payload in payloads:
        assert _xxh64_python(payload) == xxhash.xxh64_intdigest(payload), payload


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
    "members,domain",
    [([1], 1), (["a"], 2), ([], 0)],
)
def test_build_bloom_filter_encodes_value_domain(members, domain):
    blob = build_bloom_filter(members)

    assert blob[28] == domain
    assert blob[29:32] == b"\x00\x00\x00"


@pytest.mark.parametrize("fpr", [0.0001, 0.05])
def test_build_bloom_filter_accepts_fpr_boundaries(fpr):
    blob = build_bloom_filter([1], fpr=fpr)

    assert struct.unpack_from("<d", blob, 16)[0] == fpr


@pytest.mark.parametrize(
    "members,fpr",
    [
        ([1, "mixed"], 0.001),
        (["mixed", 1], 0.001),
        ([True], 0.001),
        ("not-a-list", 0.001),
        ([1 << 63], 0.001),
        ([-(1 << 63) - 1], 0.001),
        ([1], 0.00009),
        ([1], 0.051),
        ([1], float("nan")),
        ([1], float("inf")),
    ],
)
def test_build_bloom_filter_rejects_invalid_input(members, fpr):
    with pytest.raises(ParamError):
        build_bloom_filter(members, fpr=fpr)
