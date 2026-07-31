import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest
from pymilvus import BloomFilterBuilder, build_bloom_filter
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


def _scalar_reference_blob(members, fpr, domain):
    """Build the same envelope through ``_fill_scalar``, one member at a time.

    Both domains now share the vectorised numpy insert, so the scalar loop is the independent
    reference it has to agree with. The reference is assembled explicitly rather than by
    monkeypatching a seam inside the default path: a refactor that moves the vectorised code
    somewhere else then makes these tests fail loudly, instead of silently comparing the
    default path to itself (which is exactly what happened when the INT64 fill moved).
    """
    count = len(members)
    num_bytes = bloom_filter._optimal_num_bytes(count, float(fpr))
    num_blocks = num_bytes // 32
    buf = bytearray(bloom_filter._HEADER_SIZE + num_bytes)
    struct.pack_into(
        bloom_filter._HEADER_FORMAT, buf, 0, b"MBF1", 1, 1, count, float(fpr), num_blocks, domain
    )
    bloom_filter._fill_scalar(buf, members, domain, num_blocks)
    return bytes(buf)


@pytest.mark.parametrize(
    ("members", "domain"),
    [
        ([-(1 << 63), -1, 0, 1, 42, (1 << 63) - 1], bloom_filter._DOMAIN_INT64),
        (["", "a", "milvus", "日本語", "🚀🚀", "x" * 40], bloom_filter._DOMAIN_UTF8),
    ],
)
def test_build_bloom_filter_fallback_matches_default_path(monkeypatch, members, domain):
    """Forcing the pure-Python hashes must reproduce the blob the default path builds.

    Keeps the fallback exercised even when the optional C accelerator is installed. The
    vectorised INT64 path inlines its own hash and never consults ``_xxh64_int64``, so the
    load-bearing assertion is the one against the scalar reference, which does.
    """
    expected = build_bloom_filter(members, fpr=0.001)

    monkeypatch.setattr(bloom_filter, "_xxh64", bloom_filter._xxh64_python)
    monkeypatch.setattr(bloom_filter, "_xxh64_int64", bloom_filter._xxh64_int64_python)

    assert _scalar_reference_blob(members, 0.001, domain) == expected
    assert build_bloom_filter(members, fpr=0.001) == expected


@pytest.mark.parametrize("count", [1, 2, 100, (1 << 14) + 3])
def test_int64_vectorised_matches_scalar(count):
    """The numpy INT64 path must agree with the scalar loop, including across chunk boundaries."""
    rng = random.Random(20260728)
    members = [-(1 << 63), (1 << 63) - 1, 0, -1]
    members += [rng.randrange(-(1 << 63), 1 << 63) for _ in range(count)]

    assert build_bloom_filter(members, fpr=0.001) == _scalar_reference_blob(
        members, 0.001, bloom_filter._DOMAIN_INT64
    )


@pytest.mark.parametrize("count", [1, 2, 100, (1 << 14) + 3])
def test_string_vectorised_matches_scalar(count):
    """The string path now shares the vectorised insert, so it needs the same guarantee."""
    rng = random.Random(20260731)
    alphabet = "abcdefghij日本語🚀"
    members = ["", "a", "milvus", "日本語", "🚀🚀", "x" * 40]
    members += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40))) for _ in range(count)
    ]

    assert build_bloom_filter(members, fpr=0.001) == _scalar_reference_blob(
        members, 0.001, bloom_filter._DOMAIN_UTF8
    )


def test_int64_vectorised_block_words_pinned_little_endian(monkeypatch):
    """The block-word view must be explicitly little-endian, never host-native.

    ``np.uint32`` follows the host byte order: on a big-endian client (s390x, ppc64) it would
    write byte-swapped SBBF words that a little-endian QueryNode reads as different bit
    positions -- silent false negatives. The wire dtype is pinned as ``_WORD_DTYPE = "<u4"``,
    and this test proves the fill actually routes through it: forcing the constant to ">u4"
    (what native order resolves to on a big-endian host) must produce the same word *values*
    in the opposite byte order. On little-endian CI both asserts fail if the fill regresses
    to a hard-coded native dtype, because overriding the constant would then change nothing.
    """
    assert bloom_filter._WORD_DTYPE.newbyteorder("<") == bloom_filter._WORD_DTYPE

    rng = random.Random(20260729)
    members = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(500)]
    expected = build_bloom_filter(members, fpr=0.001)

    monkeypatch.setattr(bloom_filter, "_WORD_DTYPE", np.dtype(">u4"))
    swapped = build_bloom_filter(members, fpr=0.001)

    header_size = bloom_filter._HEADER_SIZE
    assert swapped[:header_size] == expected[:header_size]
    assert swapped != expected
    expected_words = np.frombuffer(expected, dtype="<u4", offset=header_size)
    swapped_words = np.frombuffer(swapped, dtype=">u4", offset=header_size)
    assert np.array_equal(swapped_words, expected_words)


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


@pytest.mark.parametrize("chunk", [1, 7, 1000, (1 << 14) + 3])
def test_builder_streaming_matches_one_shot(chunk):
    """Streaming a set through the builder must produce exactly the one-shot blob.

    Insertion into an SBBF is order-independent and idempotent, so how the members are cut
    into batches cannot matter -- including batches that straddle the internal chunk size.
    """
    ints = list(range(5000))
    strings = [f"user_{i}" for i in range(5000)]

    for members, add in (
        (ints, "add_int64_batch"),
        (strings, "add_string_batch"),
    ):
        builder = BloomFilterBuilder(len(members), fpr=0.001)
        for start in range(0, len(members), chunk):
            getattr(builder, add)(members[start : start + chunk])
        assert builder.build() == build_bloom_filter(members, fpr=0.001)


def test_builder_accepts_a_generator():
    """The point of the builder is not materialising the set, so it must take any iterable."""
    n = 40000
    builder = BloomFilterBuilder(n, fpr=0.001)
    builder.add_int64_batch(i for i in range(n))

    assert builder.build() == build_bloom_filter(list(range(n)), fpr=0.001)


def test_builder_records_both_domains():
    """A builder may mix domains -- a JSON path can legitimately hold ints and strings."""
    builder = BloomFilterBuilder(2, fpr=0.001)
    builder.add_int64_batch([1]).add_string_batch(["a"])

    blob = builder.build()
    assert blob[28] == bloom_filter._DOMAIN_INT64 | bloom_filter._DOMAIN_UTF8

    # An empty batch must not claim a domain, or the server would probe one that has no members.
    empty = BloomFilterBuilder(0, fpr=0.001)
    empty.add_int64_batch([]).add_string_batch([])
    assert empty.build()[28] == 0


def test_builder_count_only_drives_sizing():
    """n sizes the filter; overshooting it is allowed and only costs false-positive rate."""
    builder = BloomFilterBuilder(10, fpr=0.001)
    builder.add_int64_batch(range(500))
    blob = builder.build()

    # Same geometry as a 10-member filter, and n_declared still reports what was promised.
    assert len(blob) == len(build_bloom_filter(list(range(10)), fpr=0.001))
    assert struct.unpack_from("<Q", blob, 8)[0] == 10


@pytest.mark.parametrize(
    ("n", "fpr"),
    [(-1, 0.001), (1.5, 0.001), (True, 0.001), ("x", 0.001), (1, 0.0), (1, float("nan"))],
)
def test_builder_rejects_invalid_construction(n, fpr):
    with pytest.raises(ParamError):
        BloomFilterBuilder(n, fpr)


@pytest.mark.parametrize(
    ("method", "values"),
    [
        ("add_int64_batch", [1, "x"]),
        ("add_int64_batch", [True]),
        ("add_int64_batch", [1 << 63]),
        ("add_string_batch", ["a", 1]),
        ("add_string_batch", [None]),
    ],
)
def test_builder_rejects_invalid_members(method, values):
    builder = BloomFilterBuilder(len(values), fpr=0.001)
    with pytest.raises(ParamError):
        getattr(builder, method)(values)
