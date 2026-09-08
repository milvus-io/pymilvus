import struct
from contextlib import contextmanager
from itertools import islice
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Sequence, Tuple

import numpy as np

from pymilvus.exceptions import ParamError

_MAGIC = b"MRB1"
_VERSION = 1
_FORMAT_PORTABLE_ROARING64 = 1
# magic | version | format | cardinality | body_length | reserved
_HEADER_FORMAT = "<4sHHQQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

# Server-side limits, mirrored so an oversized set is rejected here instead of at the proxy.
_MAX_HIGH_CONTAINERS = 1 << 18
_MAX_DECODED_BYTES = 64 * 1024 * 1024
_MAX_BODY_BYTES = 128 * 1024 * 1024
_HIGH_CONTAINER_OVERHEAD = 128
_LOW_CONTAINER_OVERHEAD = 64

_COOKIE_NO_RUN = 12346
_COOKIE_RUN = 12347
_ARRAY_MAX_CARD = 4096
_BITMAP_BODY_BYTES = 8192
# The run-vs-bitmap tie-break compares against the *in-memory* size of a bitmap container, not
# its 8192-byte serialized body: roaring/v2's bitmapContainerSizeInBytes() is
# unsafe.Sizeof(bitmapContainer{}) + 65536/8, and that struct (an int plus a []uint64 header) is
# 32 bytes on every 64-bit platform. Reproduced verbatim so this writer emits the same bytes as
# the shipped Go reference rather than a merely-equivalent encoding -- 8192 here would flip the
# choice for every container holding 2048..2055 runs.
_SIZE_AS_BITMAP_CONTAINER = 32 + _BITMAP_BODY_BYTES
# A run-bearing Roaring32 with fewer containers than this omits the offset header entirely.
_NO_OFFSET_THRESHOLD = 4

_KIND_ARRAY = 0
_KIND_BITMAP = 1
_KIND_RUN = 2

# Every multi-byte field on the wire is little-endian. The dtypes are pinned rather than left
# native because a host-native dtype follows the machine: on a big-endian client (s390x, ppc64)
# it would emit byte-swapped container keys, offsets and array values that a little-endian
# QueryNode reads as entirely different members. test_wire_dtypes_are_pinned_little_endian
# proves the writer actually routes through these.
_U16_LE = np.dtype("<u2")
_U32_LE = np.dtype("<u4")

_SHIFT16 = np.uint64(16)
_SHIFT32 = np.uint64(32)
_ONE64 = np.uint64(1)
_LOW16_MASK = np.uint64(0xFFFF)

# Members are converted a chunk at a time so a one-shot iterable never has to be materialised as
# a Python list, which costs ~40 bytes per int against the 8 the key array needs.
_CHUNK = 1 << 16
# Floor on how many keys accumulate before they are merged into the distinct set. The real
# trigger is geometric (see add_int64_batch); this only stops small streams compacting per chunk.
_COMPACT_KEYS = 1 << 20

_TYPE_MESSAGE = "roaring bitmap members must be int"
_RANGE_MESSAGE = "roaring bitmap members must fit in signed int64"


def _screen_int_members(chunk: Sequence[Any]) -> None:
    """Reject anything np.fromiter would silently coerce.

    ``np.fromiter`` accepts where the scalar path raises: bool (an int subclass) becomes 0/1 and
    float is truncated. ``map(type, ...)`` runs at C speed and the common all-int case settles in
    one set comparison.
    """
    member_types = set(map(type, chunk))
    if member_types != {int}:
        for member_type in member_types:
            if member_type is bool or not issubclass(member_type, int):
                raise ParamError(message=_TYPE_MESSAGE)


def _to_keys(chunk: Iterable[int], size: int) -> np.ndarray:
    """Reinterpret signed members as Roaring64 keys.

    The mapping is normative: sign-extend to int64, then read the two's-complement bit pattern as
    uint64. ``view`` is that reinterpretation exactly -- ``INT8(-1)`` and ``INT64(-1)`` are the
    same member, ``0xffffffffffffffff``, never ``0xff``.
    """
    try:
        lanes = np.fromiter(chunk, dtype=np.int64, count=size)
    except OverflowError as exc:
        raise ParamError(message=_RANGE_MESSAGE) from exc
    except (TypeError, ValueError) as exc:
        raise ParamError(message=_TYPE_MESSAGE) from exc
    return lanes.view(np.uint64)


def _key_arrays(values: Iterable[int]) -> Iterator[np.ndarray]:
    """Yield the members as uint64 key arrays, converting a chunk at a time.

    A list or tuple is already measurable and is screened once, at C speed, then converted in a
    single pass. Anything else is a one-shot iterable (a generator, a DB cursor) whose length is
    unknown, so a bounded chunk has to be pulled into a list before ``np.fromiter`` can be given a
    count -- and screened per chunk, since by then the earlier members are gone.
    """
    if isinstance(values, (list, tuple)):
        _screen_int_members(values)
        if values:
            yield _to_keys(values, len(values))
        return
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, _CHUNK))
        if not chunk:
            return
        _screen_int_members(chunk)
        yield _to_keys(chunk, len(chunk))


class _Layout(NamedTuple):
    """The container decomposition of one key set, with every size already resolved."""

    keys: np.ndarray  # sorted distinct uint64 keys
    values: np.ndarray  # low 16 bits of each key, as the wire uint16 dtype
    container_start: np.ndarray  # index into keys of each low container's first value
    container_end: np.ndarray
    kind: np.ndarray  # _KIND_* per low container
    descriptors: np.ndarray  # interleaved (container key, cardinality - 1) per low container
    body_offset: np.ndarray  # exclusive prefix sum of container body sizes
    run_start: np.ndarray  # index into keys of every run's first value
    run_lo: np.ndarray  # slice of run_start owned by each low container
    run_hi: np.ndarray
    group_start: np.ndarray  # slice of the low containers owned by each high container
    group_end: np.ndarray
    high_key: np.ndarray
    container_count: np.ndarray  # low containers per high container
    has_run: np.ndarray
    has_offsets: np.ndarray
    cookie_size: np.ndarray
    run_bitmap_size: np.ndarray
    body_length: int


def _bucket_counts(keys: np.ndarray) -> Tuple[int, int]:
    """Count distinct high-32 and low-16 buckets in bounded memory.

    Chunked rather than vectorised over the whole array on purpose: the point of this function is
    to spend almost nothing, so its own temporaries have to stay small no matter how many members
    the caller passed.
    """
    if keys.size == 0:
        return 0, 0
    high_count = 1
    low_count = 1
    for start in range(1, keys.size, _CHUNK):
        stop = min(start + _CHUNK, keys.size)
        current = keys[start:stop]
        previous = keys[start - 1 : stop - 1]
        high_count += int(np.count_nonzero((current >> _SHIFT32) != (previous >> _SHIFT32)))
        low_count += int(np.count_nonzero((current >> _SHIFT16) != (previous >> _SHIFT16)))
    return high_count, low_count


def _check_bucket_limits(high_count: int, low_count: int) -> None:
    """Refuse a hopeless key set before ``_plan`` allocates anything proportional to it.

    ``_plan`` builds a dozen arrays sized by the member or container count, which for a sparse set
    -- shuffled full-range int64 ids land in nearly one high container each -- costs hundreds of
    megabytes only to conclude the server would reject it anyway. Five million random int64
    members measured at ~700 MB of transient allocation before this gate existed.

    Both limits are decidable from the counts. The high-container count is compared exactly. The
    decoded-size estimate is not yet exact -- it also includes the body length, which does not
    exist until the layout does -- but the per-container overhead alone is a *lower bound*, so a
    set whose overhead already exceeds the cap can never fit however small its body turns out to
    be. Rejecting on that lower bound is therefore sound, and the message says "at least" rather
    than quoting the bound as the size: a caller has to know what to shrink, and understating the
    figure would misinform them. Borderline sets, where the body is what tips the estimate over,
    still fall through to the exact check in ``_plan``.
    """
    if high_count > _MAX_HIGH_CONTAINERS:
        raise ParamError(
            message=f"high-container count {high_count} exceeds maximum {_MAX_HIGH_CONTAINERS}"
        )
    overhead = high_count * _HIGH_CONTAINER_OVERHEAD + low_count * _LOW_CONTAINER_OVERHEAD
    if overhead > _MAX_DECODED_BYTES:
        raise ParamError(
            message=(
                f"estimated decoded size is at least {overhead}, "
                f"exceeding maximum {_MAX_DECODED_BYTES}"
            )
        )


def _plan(keys: np.ndarray) -> _Layout:
    """Split sorted distinct keys into containers, choose each encoding and size the body.

    Everything is contiguous because the keys are sorted, so each grouping is a boundary scan
    rather than a hash: ``key >> 16`` identifies a low container (high 32 bits and the container's
    own 16 combined), and ``key >> 32`` identifies its high container.
    """
    count = keys.size

    container_id = keys >> _SHIFT16
    boundary = np.empty(count, dtype=bool)
    boundary[0] = True
    np.not_equal(container_id[1:], container_id[:-1], out=boundary[1:])
    container_start = np.flatnonzero(boundary)
    container_end = np.empty_like(container_start)
    container_end[:-1] = container_start[1:]
    container_end[-1] = count
    card = (container_end - container_start).astype(np.int64)

    # A run breaks wherever the keys stop being consecutive -- and always at a container
    # boundary, since 0x0000ffff and 0x00010000 are consecutive keys in different containers.
    run_boundary = np.empty(count, dtype=bool)
    run_boundary[0] = True
    np.not_equal(keys[1:], keys[:-1] + _ONE64, out=run_boundary[1:])
    run_boundary[1:] |= boundary[1:]
    run_start = np.flatnonzero(run_boundary)
    run_lo = np.searchsorted(run_start, container_start)
    run_hi = np.searchsorted(run_start, container_end)
    num_runs = (run_hi - run_lo).astype(np.int64)

    size_as_run = 2 + 4 * num_runs
    size_as_array = 2 * card
    kind = np.where(
        size_as_run < np.minimum(size_as_array, _SIZE_AS_BITMAP_CONTAINER),
        _KIND_RUN,
        np.where(card <= _ARRAY_MAX_CARD, _KIND_ARRAY, _KIND_BITMAP),
    )
    body_size = np.where(
        kind == _KIND_RUN,
        size_as_run,
        np.where(kind == _KIND_BITMAP, _BITMAP_BODY_BYTES, size_as_array),
    )

    container_key = container_id[container_start]
    high_of_container = container_key >> _SHIFT16
    group_boundary = np.empty(container_start.size, dtype=bool)
    group_boundary[0] = True
    np.not_equal(high_of_container[1:], high_of_container[:-1], out=group_boundary[1:])
    group_start = np.flatnonzero(group_boundary)
    group_end = np.empty_like(group_start)
    group_end[:-1] = group_start[1:]
    group_end[-1] = container_start.size
    container_count = (group_end - group_start).astype(np.int64)

    high_count = group_start.size
    if high_count > _MAX_HIGH_CONTAINERS:
        raise ParamError(
            message=f"high-container count {high_count} exceeds maximum {_MAX_HIGH_CONTAINERS}"
        )

    has_run = np.logical_or.reduceat(kind == _KIND_RUN, group_start)
    cookie_size = np.where(has_run, 4, 8)
    run_bitmap_size = np.where(has_run, (container_count + 7) // 8, 0)
    has_offsets = ~has_run | (container_count >= _NO_OFFSET_THRESHOLD)
    offset_size = np.where(has_offsets, 4 * container_count, 0)
    roaring32_size = (
        cookie_size
        + run_bitmap_size
        + 4 * container_count
        + offset_size
        + np.add.reduceat(body_size, group_start)
    )
    # 8 for the high container count, then 4 per high key.
    body_length = 8 + int(np.sum(roaring32_size)) + 4 * high_count
    if body_length > _MAX_BODY_BYTES:
        raise ParamError(message=f"body too large: {body_length}")
    estimated = (
        body_length
        + high_count * _HIGH_CONTAINER_OVERHEAD
        + container_start.size * _LOW_CONTAINER_OVERHEAD
    )
    if estimated > _MAX_DECODED_BYTES:
        raise ParamError(
            message=f"estimated decoded size {estimated} exceeds maximum {_MAX_DECODED_BYTES}"
        )

    descriptors = np.empty(2 * container_start.size, dtype=_U16_LE)
    descriptors[0::2] = container_key & _LOW16_MASK
    descriptors[1::2] = card - 1

    return _Layout(
        keys=keys,
        values=(keys & _LOW16_MASK).astype(_U16_LE),
        container_start=container_start,
        container_end=container_end,
        kind=kind,
        descriptors=descriptors,
        body_offset=np.cumsum(body_size) - body_size,
        run_start=run_start,
        run_lo=run_lo,
        run_hi=run_hi,
        group_start=group_start,
        group_end=group_end,
        high_key=high_of_container[group_start],
        container_count=container_count,
        has_run=has_run,
        has_offsets=has_offsets,
        cookie_size=cookie_size,
        run_bitmap_size=run_bitmap_size,
        body_length=body_length,
    )


def _container_body(layout: _Layout, index: int) -> bytes:
    """Serialize one low container's body: the values, the bit array, or the run list."""
    start = layout.container_start[index]
    end = layout.container_end[index]
    kind = layout.kind[index]
    if kind == _KIND_ARRAY:
        return layout.values[start:end].tobytes()
    if kind == _KIND_BITMAP:
        # Byte v>>3 bit v&7 of the packed output is word v>>6 bit v&63 of 1024 little-endian
        # uint64s, so packing a 65536-flag array LSB-first *is* the wire format -- and being a
        # byte array it carries no endianness of its own.
        flags = np.zeros(1 << 16, dtype=np.uint8)
        flags[layout.values[start:end]] = 1
        return np.packbits(flags, bitorder="little").tobytes()

    starts = layout.run_start[layout.run_lo[index] : layout.run_hi[index]]
    ends = np.empty(starts.size, dtype=np.int64)
    ends[:-1] = starts[1:] - 1
    ends[-1] = end - 1
    pairs = np.empty(2 * starts.size, dtype=_U16_LE)
    pairs[0::2] = layout.values[starts]
    pairs[1::2] = layout.keys[ends] - layout.keys[starts]
    return struct.pack("<H", starts.size) + pairs.tobytes()


def _serialize(keys: np.ndarray) -> bytes:
    """Wrap the portable Roaring64 serialization of ``keys`` in an MRB1 envelope."""
    if keys.size == 0:
        # An empty set is still a well-formed bitmap: zero high containers, an eight-byte body.
        return struct.pack(
            _HEADER_FORMAT, _MAGIC, _VERSION, _FORMAT_PORTABLE_ROARING64, 0, 8, 0
        ) + (b"\x00" * 8)

    _check_bucket_limits(*_bucket_counts(keys))
    layout = _plan(keys)
    blob = bytearray(_HEADER_SIZE + layout.body_length)
    view = memoryview(blob)
    struct.pack_into(
        _HEADER_FORMAT,
        blob,
        0,
        _MAGIC,
        _VERSION,
        _FORMAT_PORTABLE_ROARING64,
        keys.size,
        layout.body_length,
        0,
    )

    position = _HEADER_SIZE
    high_count = layout.group_start.size
    struct.pack_into("<Q", blob, position, high_count)
    position += 8

    for group in range(high_count):
        first = int(layout.group_start[group])
        last = int(layout.group_end[group])
        count = int(layout.container_count[group])
        struct.pack_into("<I", blob, position, int(layout.high_key[group]))
        position += 4

        if layout.has_run[group]:
            struct.pack_into("<HH", blob, position, _COOKIE_RUN, count - 1)
            position += 4
            run_bitmap = np.packbits(
                layout.kind[first:last] == _KIND_RUN, bitorder="little"
            ).tobytes()
            view[position : position + len(run_bitmap)] = run_bitmap
            position += len(run_bitmap)
        else:
            struct.pack_into("<II", blob, position, _COOKIE_NO_RUN, count)
            position += 8

        descriptors = layout.descriptors[2 * first : 2 * last].tobytes()
        view[position : position + len(descriptors)] = descriptors
        position += len(descriptors)

        if layout.has_offsets[group]:
            # Offsets are relative to the start of this Roaring32 blob. The base spans the cookie,
            # the run bitmap and both 4-byte-per-container headers.
            base = int(layout.cookie_size[group]) + int(layout.run_bitmap_size[group]) + 8 * count
            offsets = (layout.body_offset[first:last] - layout.body_offset[first] + base).astype(
                _U32_LE
            )
            packed = offsets.tobytes()
            view[position : position + len(packed)] = packed
            position += len(packed)

        for index in range(first, last):
            body = _container_body(layout, index)
            view[position : position + len(body)] = body
            position += len(body)

    if position != _HEADER_SIZE + layout.body_length:
        message = f"wrote {position - _HEADER_SIZE} body bytes, predicted {layout.body_length}"
        raise RuntimeError(message)
    return bytes(blob)


class RoaringBitmapBuilder:
    """Incrementally builds the blob :func:`build_roaring_bitmap` returns in one shot.

    Use this when the membership set does not comfortably fit in memory as a list of Python ints:
    a 10M-element list costs roughly 400 MB, while the keys it collapses to cost 80 MB and shrink
    further with every duplicate. Members can be streamed in from a cursor, a file or a paginated
    API and dropped as they go, in any order and with any number of repeats -- the bitmap is a
    set, so only the distinct keys survive.

    :meth:`add_int64_batch` takes an **iterable**, not one member at a time: batches are converted
    with numpy, and a per-member Python call would cost more than the conversion does. Accumulate
    into batches of a few thousand and hand those over.

    A builder is not safe for concurrent use. Overlapping calls to :meth:`add_int64_batch`,
    :meth:`build` or :attr:`cardinality` fail immediately instead of waiting. If an add raises for
    any reason, discard the builder: it is poisoned because earlier chunks of that same batch may
    already have been absorbed, and all subsequent calls will fail.

    :meth:`build` is non-destructive -- the builder may be extended and rebuilt afterwards.

    .. code-block:: python

        builder = RoaringBitmapBuilder()
        for chunk in read_ids_in_chunks(conn, size=100_000):
            builder.add_int64_batch(chunk)
        blob = builder.build()

    Build the bitmap from the SAME value domain as the target field: ``roaring_match`` matches
    INT8/INT16/INT32/INT64 fields, and a narrow member is sign-extended, so ``INT8(-1)`` and
    ``-1`` are the same member.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._failed = False
        self._keys = np.empty(0, dtype=np.uint64)
        self._parts: List[np.ndarray] = []
        self._pending = 0

    def add_int64_batch(self, values: Iterable[int]) -> "RoaringBitmapBuilder":
        """Adds integer members, poisoning the builder if the add fails."""
        with self._operation():
            try:
                for array in _key_arrays(values):
                    self._parts.append(array)
                    self._pending += array.size
                    # Geometric, not fixed: compacting every N keys re-merges the whole
                    # accumulated set every N members, which is quadratic. Waiting until the
                    # pending batches are as large as what they will be merged into makes the
                    # total work linearithmic -- 2.4x on a 10M stream.
                    if self._pending >= max(_COMPACT_KEYS, self._keys.size):
                        self._compact()
            except BaseException:
                self._failed = True
                raise
        return self

    @contextmanager
    def _operation(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            message = "RoaringBitmapBuilder does not support concurrent use"
            raise RuntimeError(message)
        try:
            if self._failed:
                message = "RoaringBitmapBuilder is unusable after a failed add"
                raise RuntimeError(message)
            yield
        finally:
            self._lock.release()

    def _compact(self) -> None:
        """Merges the pending batches into the sorted distinct key set.

        All or nothing. Every allocation that can fail happens before any state is replaced, so
        an exception -- a MemoryError out of one of these merges is the realistic one, since they
        are the largest allocations the builder makes -- leaves the pending batches exactly where
        they were and a retry still sees every member. Clearing ``_parts`` up front instead would
        drop the only reference to them, and because ``build()`` and ``cardinality`` do not poison
        the builder, the retry would quietly return a bitmap missing whatever was pending: a
        false negative in a predicate whose entire purpose is exactness.
        """
        if not self._parts:
            return
        fresh = np.unique(np.concatenate(self._parts))
        if self._keys.size == 0:
            merged_keys = fresh
        else:
            merged = np.concatenate([self._keys, fresh])
            # Both halves arrive sorted, so a stable sort is timsort merging two existing runs in
            # linear time rather than sorting from scratch -- 3x faster than np.unique over the
            # concatenation, and the result is identical either way.
            merged.sort(kind="stable")
            keep = np.empty(merged.size, dtype=bool)
            keep[0] = True
            np.not_equal(merged[1:], merged[:-1], out=keep[1:])
            merged_keys = merged[keep]

        self._keys = merged_keys
        self._parts = []
        self._pending = 0

    @property
    def cardinality(self) -> int:
        """The number of distinct members added so far. Compacts the pending batches."""
        with self._operation():
            self._compact()
            return int(self._keys.size)

    def build(self) -> bytes:
        """Serializes the members added so far into an MRB1 blob."""
        with self._operation():
            self._compact()
            return _serialize(self._keys)


def build_roaring_bitmap(members: Sequence[int]) -> bytes:
    """Build an MRB1-wrapped portable Roaring64 bitmap for roaring_match.

    Members are signed integers targeting an INT8/INT16/INT32/INT64 field, in any order and with
    any number of duplicates. The returned bytes are passed through ``filter_params``, and are
    reusable: building a large bitmap is the expensive part, so hold on to the blob and send it
    with as many queries as you like.

    For a set too large to materialise as a list, stream it through :class:`RoaringBitmapBuilder`
    instead.
    """
    if not isinstance(members, (list, tuple)):
        raise ParamError(message="roaring bitmap members must be a list or tuple of int")
    return RoaringBitmapBuilder().add_int64_batch(members).build()


def _decompose(keys: np.ndarray) -> Tuple[int, int, Dict[str, int]]:
    """Report the container shape of a key set: high count, low count and kinds.

    Not part of the wire format -- it exists so the tests can assert the container choices the
    golden fixtures document, without reparsing the blob.
    """
    if keys.size == 0:
        return 0, 0, {}
    layout = _plan(np.unique(keys))
    kinds = {}
    for name, value in (("array", _KIND_ARRAY), ("bitmap", _KIND_BITMAP), ("run", _KIND_RUN)):
        found = int(np.count_nonzero(layout.kind == value))
        if found:
            kinds[name] = found
    return layout.group_start.size, layout.container_start.size, kinds
