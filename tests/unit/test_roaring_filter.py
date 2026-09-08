import base64
import contextlib
import hashlib
import inspect
import json
import random
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from pymilvus import MilvusClient, RoaringBitmapBuilder, build_roaring_bitmap
from pymilvus.client import roaring_filter
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import MilvusException, ParamError

MASK64 = (1 << 64) - 1

GOLDEN_PATH = Path(__file__).parent / "testdata" / "roaring_golden_vectors.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())
CASES = GOLDEN["cases"]
CASE_IDS = [case["name"] for case in CASES]


def expand(case):
    """Expand a fixture's run-length member spans into the member list."""
    members = []
    for span in case["members"]:
        start, step = int(span["start"]), int(span["step"])
        members.extend(start + index * step for index in range(span["count"]))
    return members


# ---------------------------------------------------------------------------------------------
# An independent reader, used only by the tests. The SDK never parses a blob -- proxy and
# QueryNode do -- so byte equality against the golden vectors is the primary conformance signal.
# This reader is the secondary one: it makes the randomised tests meaningful (they have no golden
# bytes to compare against) and it checks the offset table, which byte equality would only catch
# for the shapes the fixtures happen to cover. It is validated against the fixture bytes by
# test_reader_recovers_golden_members before anything else relies on it.
# ---------------------------------------------------------------------------------------------


def read_roaring32(blob, start):
    position = start
    (cookie16,) = struct.unpack_from("<H", blob, position)
    if cookie16 == 12347:
        _, minus_one = struct.unpack_from("<HH", blob, position)
        position += 4
        count = minus_one + 1
        bitmap_bytes = (count + 7) // 8
        run_bits = blob[position : position + bitmap_bytes]
        position += bitmap_bytes
        is_run = [bool(run_bits[i // 8] >> (i % 8) & 1) for i in range(count)]
        # Trailing pad bits must be zero.
        assert int.from_bytes(run_bits, "little") >> count == 0
    else:
        cookie, count = struct.unpack_from("<II", blob, position)
        position += 8
        assert cookie == 12346, cookie
        is_run = [False] * count

    keys, cards = [], []
    for _ in range(count):
        key, card_minus_one = struct.unpack_from("<HH", blob, position)
        position += 4
        keys.append(key)
        cards.append(card_minus_one + 1)
    assert keys == sorted(set(keys)), "container keys must be ascending and distinct"

    offsets = None
    if not any(is_run) or count >= 4:
        offsets = struct.unpack_from(f"<{count}I", blob, position)
        position += 4 * count

    values = []
    for index in range(count):
        if offsets is not None:
            assert offsets[index] == position - start, (index, offsets[index], position - start)
        high = keys[index] << 16
        if is_run[index]:
            (runs,) = struct.unpack_from("<H", blob, position)
            position += 2
            for _ in range(runs):
                first, length_minus_one = struct.unpack_from("<HH", blob, position)
                position += 4
                values.extend(high | (first + step) for step in range(length_minus_one + 1))
        elif cards[index] > 4096:
            words = struct.unpack_from("<1024Q", blob, position)
            position += 8192
            values.extend(
                high | (word << 6 | bit)
                for word in range(1024)
                for bit in range(64)
                if words[word] >> bit & 1
            )
        else:
            packed = struct.unpack_from(f"<{cards[index]}H", blob, position)
            position += 2 * cards[index]
            values.extend(high | value for value in packed)
        assert len(values) == sum(cards[: index + 1]), "container cardinality disagrees with body"
    return position, values


def read_roaring(blob):
    """Return the members a blob encodes, as unsigned keys, checking the envelope on the way."""
    magic, version, fmt, cardinality, body_length, reserved = struct.unpack_from(
        "<4sHHQQQ", blob, 0
    )
    assert magic == b"MRB1"
    assert (version, fmt, reserved) == (1, 1, 0)
    assert body_length == len(blob) - 32
    position = 32
    (high_count,) = struct.unpack_from("<Q", blob, position)
    position += 8

    members, highs = [], []
    for _ in range(high_count):
        (high,) = struct.unpack_from("<I", blob, position)
        position += 4
        highs.append(high)
        position, values = read_roaring32(blob, position)
        members.extend(high << 32 | value for value in values)
    assert highs == sorted(set(highs)), "high keys must be ascending and distinct"
    assert position == len(blob), (position, len(blob))
    assert len(members) == cardinality
    assert members == sorted(set(members)), "members must come out ascending and distinct"
    return members


def test_golden_vector_file_is_the_expected_suite():
    assert GOLDEN["spec"] == "MRB1"
    assert GOLDEN["version"] == 1
    assert len(CASES) == 29
    kinds = {}
    for case in CASES:
        for kind, found in case["container_kinds"].items():
            kinds[kind] = kinds.get(kind, 0) + found
    assert kinds == {"array": 47, "bitmap": 4, "run": 20}


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_build_roaring_bitmap_matches_golden_vector(case):
    """Byte-for-byte equality with the shipped Go reference, for every fixture."""
    members = expand(case)
    assert len(members) == case["member_count"]

    blob = build_roaring_bitmap(members)
    expected = base64.b64decode(case["blob_base64"])
    if blob != expected:
        shared = min(len(blob), len(expected))
        offset = next((i for i in range(shared) if blob[i] != expected[i]), shared)
        pytest.fail(
            f"{case['name']}: built {len(blob)} bytes, expected {len(expected)}; "
            f"first difference at offset {offset}: "
            f"got {blob[offset : offset + 16].hex()} want {expected[offset : offset + 16].hex()}"
        )
    assert hashlib.sha256(blob).hexdigest() == case["blob_sha256"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_golden_vector_header_and_shape(case):
    """The header and the container decomposition must match what the fixture documents."""
    members = expand(case)
    blob = build_roaring_bitmap(members)

    magic, version, fmt, cardinality, body_length, reserved = struct.unpack_from(
        "<4sHHQQQ", blob, 0
    )
    assert magic == b"MRB1"
    assert (version, fmt, reserved) == (1, 1, 0)
    assert cardinality == case["cardinality"]
    assert body_length == case["body_length"]
    assert len(blob) == 32 + body_length

    keys = np.fromiter(members, dtype=np.int64, count=len(members)).view(np.uint64)
    high, low, kinds = roaring_filter._decompose(keys)
    assert high == case["high_container_count"]
    assert low == case["low_container_count"]
    assert kinds == case["container_kinds"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_reader_recovers_golden_members(case):
    """Validates the test-only reader against known-good bytes, before the random tests use it."""
    expected = base64.b64decode(case["blob_base64"])

    assert read_roaring(expected) == sorted({member & MASK64 for member in expand(case)})


# ---------------------------------------------------------------------------------------------
# Signed mapping
# ---------------------------------------------------------------------------------------------


def single_member_blob(key):
    """Hand-assemble the blob for one member, from the spec rather than from the builder.

    A single member is always one high container holding one single-value ARRAY container, so the
    whole encoding is fixed: cookie, one descriptor, one offset (which lands right after the
    cookie plus both 4-byte-per-container headers) and the two-byte value.
    """
    body = struct.pack("<Q", 1)
    body += struct.pack("<I", key >> 32)
    body += struct.pack("<II", 12346, 1)
    body += struct.pack("<HH", key >> 16 & 0xFFFF, 0)
    body += struct.pack("<I", 8 + 8)
    body += struct.pack("<H", key & 0xFFFF)
    return struct.pack("<4sHHQQQ", b"MRB1", 1, 1, 1, len(body), 0) + body


@pytest.mark.parametrize(
    ("value", "key"),
    [
        (-1, 0xFFFFFFFFFFFFFFFF),
        (-128, 0xFFFFFFFFFFFFFF80),
        (-32768, 0xFFFFFFFFFFFF8000),
        (-2147483648, 0xFFFFFFFF80000000),
        (-(1 << 63), 0x8000000000000000),
        (0, 0x0000000000000000),
        (42, 0x000000000000002A),
        ((1 << 63) - 1, 0x7FFFFFFFFFFFFFFF),
    ],
)
def test_signed_members_use_twos_complement_keys(value, key):
    """A signed member is sign-extended to int64 and its bit pattern read as the uint64 key."""
    assert build_roaring_bitmap([value]) == single_member_blob(key)


@pytest.mark.parametrize(
    ("narrow", "wrong_key"),
    [(-1, 0xFF), (-128, 0x80), (-32768, 0x8000), (-2147483648, 0x80000000)],
)
def test_narrow_negative_members_are_not_zero_extended(narrow, wrong_key):
    """INT8(-1) is the member -1, not the member 255. Zero extension is the classic bug here."""
    assert build_roaring_bitmap([narrow]) != single_member_blob(wrong_key)
    assert build_roaring_bitmap([narrow]) == build_roaring_bitmap([narrow, narrow])


def test_ordering_is_unsigned():
    """{-1, 5} sorts as {5, 0xffff...} -- the high container for -1 comes last, not first."""
    blob = build_roaring_bitmap([-1, 5])

    assert read_roaring(blob) == [5, MASK64]
    # High key 0 is written before high key 0xffffffff.
    assert struct.unpack_from("<I", blob, 40)[0] == 0


@pytest.mark.parametrize(
    "members",
    [
        [-(1 << 63), -1, 0, 1, 42, (1 << 63) - 1],
        [(1 << 63) - 1, -(1 << 63)],
        [-1, 0],
        [0, -1],
    ],
)
def test_signed_members_round_trip(members):
    assert read_roaring(build_roaring_bitmap(members)) == sorted(
        {member & MASK64 for member in members}
    )


# ---------------------------------------------------------------------------------------------
# Set semantics
# ---------------------------------------------------------------------------------------------


def test_duplicates_collapse_and_order_does_not_matter():
    """The bitmap is a set: the same members in any order and multiplicity give the same bytes."""
    distinct = [-(1 << 63), -1, 0, 5, 6, 7, 1 << 40, (1 << 40) + 1, (1 << 63) - 1]
    expected = build_roaring_bitmap(distinct)
    assert struct.unpack_from("<Q", expected, 8)[0] == len(distinct)

    rng = random.Random(20260817)
    for _ in range(25):
        shuffled = distinct * 3
        rng.shuffle(shuffled)
        assert build_roaring_bitmap(shuffled) == expected
        assert build_roaring_bitmap(tuple(shuffled)) == expected


def test_empty_member_set():
    """An empty set is still well formed: zero high containers, an eight-byte body."""
    blob = build_roaring_bitmap([])

    assert len(blob) == 40
    assert blob[:4] == b"MRB1"
    assert struct.unpack_from("<Q", blob, 8)[0] == 0
    assert struct.unpack_from("<Q", blob, 16)[0] == 8
    assert blob[32:] == b"\x00" * 8
    assert read_roaring(blob) == []
    assert build_roaring_bitmap(()) == blob
    assert RoaringBitmapBuilder().build() == blob
    assert RoaringBitmapBuilder().add_int64_batch([]).build() == blob


@pytest.mark.parametrize("seed", range(12))
def test_random_member_sets_round_trip(seed):
    """Randomised shapes that bracket every container-selection boundary must decode exactly.

    Byte equality only covers the 29 fixture shapes; this covers the space between them, and the
    reader checks the offset table and every declared cardinality as it goes.
    """
    rng = random.Random(20260817 + seed)
    members = []
    for _ in range(rng.randrange(1, 6)):
        high = rng.choice([0, -1, rng.randrange(-(1 << 31), 1 << 31)]) << 32
        low = rng.choice([0, rng.randrange(0, 1 << 16)]) << 16
        base = high | low
        shape = rng.choice(["sparse", "dense", "runs", "boundary"])
        if shape == "sparse":
            members.extend(base + rng.randrange(0, 1 << 16) for _ in range(rng.randrange(1, 200)))
        elif shape == "dense":
            members.extend(base + value for value in range(rng.randrange(4000, 6200)))
        elif shape == "runs":
            start = 0
            for _ in range(rng.randrange(1, 2100)):
                length = rng.randrange(1, 4)
                members.extend(base + start + step for step in range(length))
                start += length + rng.randrange(1, 4)
                if start >= (1 << 16) - 8:
                    break
        else:
            members.extend(base + value for value in range(rng.choice([4095, 4096, 4097])))
    members = [(member + (1 << 63)) % (1 << 64) - (1 << 63) for member in members]
    rng.shuffle(members)

    blob = build_roaring_bitmap(members)

    assert read_roaring(blob) == sorted({member & MASK64 for member in members})


# ---------------------------------------------------------------------------------------------
# Container selection
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_runs", "kind"),
    [(2054, "run"), (2055, "run"), (2056, "bitmap"), (2057, "bitmap")],
)
def test_run_versus_bitmap_tie_break_uses_the_in_memory_size(num_runs, kind):
    """The tie-break is against 8224, the Go bitmapContainer's in-memory size, not 8192.

    A container of N runs serializes to 2 + 4N bytes, so 2048..2055 runs is exactly the window
    where 8192 and 8224 disagree. At 2055 runs the RUN encoding is 8222 bytes -- larger on the
    wire than the 8192-byte bitmap it beats -- and that is correct: the comparison is on the
    in-memory size the reference uses.

    Each run is three values wide so the array encoding stays the largest of the three and drops
    out of the comparison; two-wide runs make the array smaller than the run and no choice of
    8192 or 8224 would ever be consulted.
    """
    members = [value for run in range(num_runs) for value in (4 * run, 4 * run + 1, 4 * run + 2)]

    assert roaring_filter._decompose(np.array(members, dtype=np.uint64))[2] == {kind: 1}


@pytest.mark.parametrize(
    ("card", "kind"),
    [(4095, "array"), (4096, "array"), (4097, "bitmap")],
)
def test_array_versus_bitmap_boundary(card, kind):
    """Above 4096 distinct values a non-run container becomes a BITMAP."""
    members = [2 * value for value in range(card)]

    assert roaring_filter._decompose(np.array(members, dtype=np.uint64))[2] == {kind: 1}


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_offset_table_is_omitted_below_four_run_containers(count):
    """A run-bearing Roaring32 writes the offset header only from four containers up."""
    members = [(container << 16) + value for container in range(count) for value in range(64)]

    blob = build_roaring_bitmap(members)
    # cookie(4) + run bitmap + 4 per descriptor, then optionally 4 per offset.
    expected_header = 4 + (count + 7) // 8 + 4 * count + (4 * count if count >= 4 else 0)
    body_start = 32 + 8 + 4
    assert struct.unpack_from("<H", blob, body_start)[0] == 12347
    assert len(blob) == body_start + expected_header + count * (2 + 4)
    assert read_roaring(blob) == sorted(members)


def test_no_run_container_always_writes_the_offset_table():
    """The threshold is a run-only rule: a cookie-12346 Roaring32 always carries offsets."""
    members = [0, 2]

    blob = build_roaring_bitmap(members)

    assert struct.unpack_from("<I", blob, 32 + 8 + 4)[0] == 12346
    # cookie(8) + descriptor(4) + offset(4) + two uint16 values.
    assert len(blob) == 32 + 8 + 4 + 8 + 4 + 4 + 4


# ---------------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------------


def test_rejects_too_many_high_containers():
    """262144 high containers is the ceiling; one more is refused, not truncated."""
    ceiling = 1 << 18
    members = [value << 32 for value in range(ceiling)]
    assert struct.unpack_from("<Q", build_roaring_bitmap(members), 32)[0] == ceiling

    with pytest.raises(ParamError, match="high-container count 262145 exceeds maximum 262144"):
        build_roaring_bitmap([*members, ceiling << 32])


def test_rejects_oversized_estimated_decoded_size(monkeypatch):
    """The decoded-size estimate counts per-container overhead, not just the body.

    200K high containers of four low containers each is 76.8 MiB of fixed overhead before a
    single body byte, so it cannot fit however the bodies encode. That verdict is decidable from
    the bucket counts, so it is reached without ``_plan`` laying out 800K containers first, and
    the message says "at least" -- quoting the overhead as if it were the size would understate
    what the caller has to shrink.
    """
    planned = []
    monkeypatch.setattr(
        roaring_filter, "_plan", lambda *args: planned.append(1) or pytest.fail("planned")
    )
    members = [high << 32 | low << 16 for high in range(200_000) for low in range(4)]

    with pytest.raises(
        ParamError, match=r"estimated decoded size is at least 76800000, exceeding maximum 67108864"
    ):
        build_roaring_bitmap(members)
    assert planned == []


def test_limits_are_checked_before_the_body_is_allocated(monkeypatch):
    """An oversized input must fail fast, not after materialising tens of MiB of bytes."""
    allocated = []
    real_serialize = roaring_filter._container_body
    monkeypatch.setattr(
        roaring_filter,
        "_container_body",
        lambda *args: allocated.append(1) or real_serialize(*args),
    )

    with pytest.raises(ParamError, match="high-container count"):
        build_roaring_bitmap([value << 32 for value in range((1 << 18) + 1)])
    assert allocated == []


def test_a_hopelessly_sparse_set_is_refused_without_planning_it(monkeypatch):
    """The layout is never built for a set the high-container limit already rules out.

    ``_plan`` allocates a dozen arrays sized by the member and container counts, so a sparse set
    -- shuffled full-range int64 ids land in nearly one high container each -- used to cost
    hundreds of megabytes of numpy temporaries only to be refused at the end of it. Asserting
    ``_plan`` is not entered is the honest form of that claim; measuring RSS would be flaky.
    """
    planned = []
    monkeypatch.setattr(
        roaring_filter, "_plan", lambda *args: planned.append(1) or pytest.fail("planned")
    )

    with pytest.raises(ParamError, match="high-container count 262145 exceeds maximum 262144"):
        build_roaring_bitmap([value << 32 for value in range((1 << 18) + 1)])
    assert planned == []


def test_rejects_oversized_body(monkeypatch):
    """The 128 MiB body cap is reproduced from the reference even though the decoded-size cap
    bites first for every reachable input -- both are checked, in the reference's order."""
    monkeypatch.setattr(roaring_filter, "_MAX_BODY_BYTES", 8)

    with pytest.raises(ParamError, match="body too large: 30"):
        build_roaring_bitmap([1])


def test_a_borderline_estimate_is_decided_by_the_planner_with_the_exact_figure(monkeypatch):
    """When only the body tips the estimate over, the planner decides and quotes the real size.

    One member costs 192 bytes of container overhead and 30 of body, so a 200-byte cap is above
    what the bucket counts can rule out but below the true 222. The early gate must let this
    through rather than refuse it on the overhead alone, and the error the caller sees must be
    the exact figure -- that is the number they have to size against.
    """
    monkeypatch.setattr(roaring_filter, "_MAX_DECODED_BYTES", 200)
    planned = []
    real_plan = roaring_filter._plan
    monkeypatch.setattr(
        roaring_filter, "_plan", lambda *args: planned.append(1) or real_plan(*args)
    )

    with pytest.raises(ParamError, match="estimated decoded size 222 exceeds maximum 200"):
        build_roaring_bitmap([1])
    assert planned == [1]


# ---------------------------------------------------------------------------------------------
# Input rejection
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "members",
    [
        [1, True],
        [True, 1],
        [True],
        [False],
        [1, "x"],
        ["1"],
        [1, None],
        [None],
        [1, 1.5],
        [1.0],
        [1, 1 << 63],
        [1 << 63],
        [-(1 << 63) - 1],
        [1, [2]],
        [b"\x01"],
    ],
)
def test_rejects_invalid_members(members):
    with pytest.raises(ParamError):
        build_roaring_bitmap(members)


@pytest.mark.parametrize("members", ["not-a-list", 42, None, {1, 2}, {1: 2}])
def test_rejects_non_sequence_members(members):
    with pytest.raises(ParamError, match="must be a list or tuple"):
        build_roaring_bitmap(members)


def test_generator_members_must_go_through_the_builder():
    """The one-shot builder needs a measurable sequence; a generator is the builder's job."""
    with pytest.raises(ParamError, match="must be a list or tuple"):
        build_roaring_bitmap(value for value in range(4))

    assert RoaringBitmapBuilder().add_int64_batch(
        value for value in range(4)
    ).build() == build_roaring_bitmap([0, 1, 2, 3])


def test_int_subclasses_are_accepted():
    """An IntEnum or a numpy-free int subclass is still an int member."""

    class Identifier(int):
        pass

    assert build_roaring_bitmap([Identifier(7), Identifier(-1)]) == build_roaring_bitmap([7, -1])


# ---------------------------------------------------------------------------------------------
# Wire dtypes
# ---------------------------------------------------------------------------------------------


def test_wire_dtypes_are_pinned_little_endian(monkeypatch):
    """Every numpy-written field must go through an explicitly little-endian dtype.

    A host-native dtype follows the machine: on a big-endian client (s390x, ppc64) it would emit
    byte-swapped container keys, offsets and array values, and a little-endian QueryNode would
    read an entirely different member set -- silent wrong answers, not an error. Forcing the
    constants to what native order resolves to on a big-endian host must byte-swap exactly those
    fields and nothing else. On little-endian CI the asserts also fail if the writer regresses to
    a hard-coded native dtype, because overriding the constants would then change nothing.
    """
    expected = build_roaring_bitmap([0x1234])
    monkeypatch.setattr(roaring_filter, "_U16_LE", np.dtype(">u2"))
    monkeypatch.setattr(roaring_filter, "_U32_LE", np.dtype(">u4"))
    swapped = build_roaring_bitmap([0x1234])

    assert len(swapped) == len(expected) == 62
    # Header, high container count, high key and cookie all go through struct and are unaffected.
    assert swapped[:52] == expected[:52]
    for start, stop, wire in ((52, 56, "u2"), (56, 60, "u4"), (60, 62, "u2")):
        assert (
            np.frombuffer(swapped[start:stop], dtype=">" + wire).tolist()
            == np.frombuffer(expected[start:stop], dtype="<" + wire).tolist()
        )
    assert swapped != expected


def test_wire_dtype_declarations_pin_little_endian_in_the_source():
    """Guards the one form of this bug no runtime assertion can see on a little-endian host.

    ``np.dtype("u2")`` and ``np.dtype("<u2")`` are the *same object* here -- numpy resolves
    native order eagerly -- so dropping the ``<`` is invisible to every runtime check, including
    the monkeypatch above, and misbehaves only on the big-endian client no CI runs on. The
    declaration itself is the only thing left to assert on.
    """
    declarations = re.findall(r"np\.dtype\(\"([^\"]+)\"\)", inspect.getsource(roaring_filter))

    assert declarations, "the wire dtypes moved or were renamed; update this test"
    for spec in declarations:
        assert spec.startswith("<"), f'np.dtype("{spec}") must pin little-endian byte order'


def test_bitmap_container_body_carries_no_endianness(monkeypatch):
    """The bitmap body is packed as bytes, so it is identical on either host.

    Byte v>>3 bit v&7 of the packed output is word v>>6 bit v&63 of 1024 little-endian uint64s.
    Writing it through a uint64 view instead would make it host-dependent.
    """
    # Every other value: 5000 single-value runs, too many to encode as runs and too many to
    # encode as an array, which is the only way to reach a BITMAP container.
    members = [2 * value for value in range(5000)]
    body_start = 32 + 8 + 4 + 8 + 4 + 4
    expected = build_roaring_bitmap(members)
    assert len(expected) == body_start + 8192

    monkeypatch.setattr(roaring_filter, "_U16_LE", np.dtype(">u2"))
    monkeypatch.setattr(roaring_filter, "_U32_LE", np.dtype(">u4"))
    swapped = build_roaring_bitmap(members)

    assert swapped[body_start:] == expected[body_start:]
    assert swapped[:body_start] != expected[:body_start]


# ---------------------------------------------------------------------------------------------
# Streaming builder
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 1000, (1 << 16) + 3])
def test_builder_streaming_matches_one_shot(chunk):
    """How the members are cut into batches cannot matter -- the bitmap is a set."""
    members = [*range(5000), *range(-3000, -1000), *(value << 32 for value in range(50))]

    builder = RoaringBitmapBuilder()
    for start in range(0, len(members), chunk):
        builder.add_int64_batch(members[start : start + chunk])

    assert builder.build() == build_roaring_bitmap(members)


def test_builder_accepts_a_generator():
    """The point of the builder is not materialising the set, so it must take any iterable."""
    count = (1 << 16) + 5
    builder = RoaringBitmapBuilder()
    builder.add_int64_batch(value for value in range(count))

    assert builder.build() == build_roaring_bitmap(list(range(count)))


def test_builder_compaction_is_not_lossy(monkeypatch):
    """Merging pending batches mid-stream must neither drop members nor resurrect duplicates.

    Compaction merges an already-sorted distinct set with a freshly sorted one; getting the
    dedup boundary wrong there is invisible unless the stream is long enough to compact
    repeatedly, so both thresholds are forced down until it does.
    """
    monkeypatch.setattr(roaring_filter, "_COMPACT_KEYS", 8)
    monkeypatch.setattr(roaring_filter, "_CHUNK", 5)
    compactions = []
    real_compact = RoaringBitmapBuilder._compact
    monkeypatch.setattr(
        RoaringBitmapBuilder,
        "_compact",
        lambda self: compactions.append(1) or real_compact(self),
    )
    members = [-(1 << 63), *range(3000), *range(2500, 3500), (1 << 63) - 1]

    builder = RoaringBitmapBuilder()
    builder.add_int64_batch(iter(members))

    assert len(compactions) > 1, "the test has to compact mid-stream to mean anything"
    assert builder.build() == build_roaring_bitmap(members)


def test_a_failed_compaction_keeps_the_pending_members(monkeypatch):
    """A merge that runs out of memory must not swallow the members it was merging.

    Compaction merges the pending batches into the accumulated key set. Those merges are the
    largest allocations the builder makes, so MemoryError is where they realistically fail, and
    neither build() nor cardinality poisons the builder afterwards. If the pending batches were
    released before the merge completed, the retry would succeed and quietly return a bitmap
    short of whatever was pending -- a false negative in a predicate whose whole point is being
    exact, and one that reports no error at all.
    """
    builder = RoaringBitmapBuilder()
    builder.add_int64_batch([1])
    builder.build()  # compacts, so {1} is in the accumulated set and nothing is pending
    builder.add_int64_batch([2])

    real_concatenate = np.concatenate
    calls = []

    def failing_concatenate(arrays, *args, **kwargs):
        calls.append(1)
        # The first call gathers the pending batches; the second merges them into _keys. Fail
        # the merge, which is the one holding the only reference to the pending members.
        if len(calls) == 2:
            raise MemoryError("injected")
        return real_concatenate(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "concatenate", failing_concatenate)
    with pytest.raises(MemoryError):
        builder.build()
    monkeypatch.undo()

    assert calls == [1, 1], "the test has to fail the merge, not the gather"
    assert builder.cardinality == 2
    assert builder.build() == build_roaring_bitmap([1, 2])


def test_builder_cardinality_counts_distinct_members():
    builder = RoaringBitmapBuilder()
    assert builder.cardinality == 0

    builder.add_int64_batch([1, 1, 2]).add_int64_batch([2, -1])
    assert builder.cardinality == 3
    assert struct.unpack_from("<Q", builder.build(), 8)[0] == 3


def test_builder_can_be_extended_after_build():
    """build() is non-destructive: a bitmap is expensive to make and worth growing in place."""
    builder = RoaringBitmapBuilder().add_int64_batch([1, 2])
    first = builder.build()

    builder.add_int64_batch([3])

    assert first == build_roaring_bitmap([1, 2])
    assert builder.build() == build_roaring_bitmap([1, 2, 3])


def test_builder_is_poisoned_when_stream_fails():
    """A failed streamed add may already have absorbed chunks, so the builder must not be used."""
    values = list(range(1 << 16))

    def failing_cursor():
        yield from values
        raise RuntimeError("cursor failed")

    builder = RoaringBitmapBuilder()
    with pytest.raises(RuntimeError, match="cursor failed"):
        builder.add_int64_batch(failing_cursor())

    with pytest.raises(RuntimeError, match="unusable after a failed add"):
        builder.add_int64_batch([1])
    with pytest.raises(RuntimeError, match="unusable after a failed add"):
        builder.build()
    with pytest.raises(RuntimeError, match="unusable after a failed add"):
        builder.cardinality  # noqa: B018


@pytest.mark.parametrize("values", [[1, "x"], [True], [1 << 63], [None]])
def test_builder_rejects_invalid_members(values):
    builder = RoaringBitmapBuilder()
    with pytest.raises(ParamError):
        builder.add_int64_batch(values)
    with pytest.raises(RuntimeError, match="unusable after a failed add"):
        builder.build()


@pytest.mark.parametrize("operation", ["add_int64_batch", "build", "cardinality"])
def test_builder_rejects_concurrent_use(operation):
    """Concurrent use is unsupported and must fail instead of corrupting the pending batches."""
    builder = RoaringBitmapBuilder()
    add_entered = Event()
    allow_add = Event()

    def blocking_values():
        add_entered.set()
        allow_add.wait()
        yield 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        add_future = pool.submit(builder.add_int64_batch, blocking_values())
        assert add_entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="does not support concurrent use"):
                if operation == "add_int64_batch":
                    builder.add_int64_batch([2])
                elif operation == "build":
                    builder.build()
                else:
                    builder.cardinality  # noqa: B018
        finally:
            allow_add.set()
        add_future.result()

    # The rejected call does not poison the active operation or the builder.
    builder.add_int64_batch([2])
    assert builder.build() == build_roaring_bitmap([1, 2])


# ---------------------------------------------------------------------------------------------
# Template plumbing
# ---------------------------------------------------------------------------------------------


def test_build_roaring_bitmap_and_template_bytes_value():
    blob = build_roaring_bitmap([1, -1, 1 << 40])

    values = Prepare.prepare_expression_template({"rb": blob})

    assert values["rb"].bytes_val == blob
    assert values["rb"].WhichOneof("val") == "bytes_val"


def test_roaring_blob_reaches_the_wire_through_the_public_client(mock_grpc_handler, mock_grpc_stub):
    """The blob survives the real client route, not just protobuf preparation.

    The request-building tests below call ``Prepare`` directly, which proves the protobuf is
    assembled correctly but steps over everything MilvusClient does on the way there. This drives
    the public query/search/delete APIs with only the gRPC stub mocked, so the whole client path
    runs, and asserts on the request that actually left for the wire.

    The mocked stub answers with an empty response, which the client rightly rejects while
    parsing; that happens after the request is sent, and the request is what is under test here.
    """
    blob = build_roaring_bitmap([1, -1, 1 << 40])
    expr = "roaring_match(id, {rb})"

    with patch("pymilvus.orm.connections.GrpcHandler", return_value=mock_grpc_handler), patch(
        "pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_grpc_handler
    ):
        client = MilvusClient()
        with contextlib.suppress(MilvusException):
            client.query("coll", filter=expr, filter_params={"rb": blob})
        with contextlib.suppress(MilvusException):
            client.search(
                "coll",
                data=[[1.0, 2.0]],
                anns_field="vector",
                filter=expr,
                filter_params={"rb": blob},
                search_params={"metric_type": "L2"},
            )
        with contextlib.suppress(MilvusException):
            client.delete("coll", filter=expr, filter_params={"rb": blob})

    sent = {
        "query": mock_grpc_stub.Query.call_args,
        "search": mock_grpc_stub.Search.call_args,
        "delete": mock_grpc_stub.Delete.future.call_args,
    }
    for route, call in sent.items():
        assert call is not None, f"the client never sent a {route} request"
        request = call[0][0]
        value = request.expr_template_values["rb"]
        assert value.WhichOneof("val") == "bytes_val", route
        assert value.bytes_val == blob, route


def test_roaring_blob_reaches_query_search_and_delete_requests():
    """The blob must arrive as protobuf bytes -- not base64, not a string -- on every path."""
    blob = build_roaring_bitmap([1, -1, 1 << 40])
    expr = "roaring_match(id, {rb})"

    query = Prepare.query_request("coll", expr, [], None, expr_params={"rb": blob})
    delete = Prepare.delete_request("coll", expr, None, 0, expr_params={"rb": blob})
    search = Prepare.search_requests_with_expr(
        collection_name="coll",
        data=[[1.0, 2.0]],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        expr=expr,
        expr_params={"rb": blob},
    )

    for request in (query, delete, search):
        assert request.expr_template_values["rb"].bytes_val == blob
        assert request.expr_template_values["rb"].WhichOneof("val") == "bytes_val"
        # Survives a protobuf round trip unchanged -- no string coercion anywhere in between.
        restored = type(request).FromString(request.SerializeToString())
        assert restored.expr_template_values["rb"].bytes_val == blob
