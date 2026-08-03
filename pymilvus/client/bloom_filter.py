import math
import struct
from itertools import islice
from threading import Lock
from typing import Any, Callable, Iterable, Iterator, Sequence, Tuple, Union

import numpy as np

from pymilvus.exceptions import ParamError

_MASK64 = (1 << 64) - 1
_PRIME64_1 = 11400714785074694791
_PRIME64_2 = 14029467366897019727
_PRIME64_3 = 1609587929392839161
_PRIME64_4 = 9650029242287828579
_PRIME64_5 = 2870177450012600261
_SALT = (
    0x47B6137B,
    0x44974D91,
    0x8824AD5B,
    0xA2B7289D,
    0x705495C7,
    0x2DF1424B,
    0x9EFC4947,
    0x5C6BFB31,
)

_MIN_FPR = 0.0001
_MAX_FPR = 0.05
_DEFAULT_FPR = 0.005
_MIN_FILTER_BYTES = 32
_MAX_FILTER_BYTES = 128 * 1024 * 1024
_DOMAIN_INT64 = 1
_DOMAIN_UTF8 = 2

_HEADER_FORMAT = "<4sHHQdIB3x"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
# One SBBF block is eight little-endian uint32 words; read and write it in a single call so the
# hot loop does two struct operations per member instead of sixteen.
_BLOCK_STRUCT = struct.Struct("<8I")
# The body is also viewed 64 bits at a time so one scatter covers two of a block's eight
# words. The dtype is pinned little-endian: a host-native dtype would emit byte-swapped lanes
# on a big-endian client.
_WORD64_DTYPE = np.dtype("<u8")
# A 32-byte block is four uint64 lanes.
_LANES_PER_BLOCK = 4
# Buffered retries before falling back to the unbuffered scatter. Three covers the collision
# rate a well-sized filter produces; the cap is what stops an oversubscribed one degenerating.
_MAX_BUFFERED_PASSES = 3


def _rotl64(value: int, count: int) -> int:
    return ((value << count) | (value >> (64 - count))) & _MASK64


def _round(accumulator: int, value: int) -> int:
    accumulator = (accumulator + value * _PRIME64_2) & _MASK64
    accumulator = _rotl64(accumulator, 31)
    return (accumulator * _PRIME64_1) & _MASK64


def _merge_round(accumulator: int, value: int) -> int:
    accumulator ^= _round(0, value)
    return (accumulator * _PRIME64_1 + _PRIME64_4) & _MASK64


def _avalanche(result: int) -> int:
    result ^= result >> 33
    result = (result * _PRIME64_2) & _MASK64
    result ^= result >> 29
    result = (result * _PRIME64_3) & _MASK64
    return result ^ (result >> 32)


def _xxh64_int64_python(value: int) -> int:
    """Return XXH64(struct.pack("<q", value), seed=0) without materialising the 8 bytes.

    Specialisation of :func:`_xxh64_python` for the one input length the INT64 domain ever
    produces: the 32-byte stripe loop and both tail loops are unreachable, leaving a single
    8-byte lane. ``value & _MASK64`` is exactly the little-endian two's-complement encoding
    read back as ``<Q``, so the pack/unpack round trip drops out. Kept honest by
    ``test_xxh64_int64_python_matches_generic``.
    """
    accumulator = ((value & _MASK64) * _PRIME64_2) & _MASK64
    accumulator = ((accumulator << 31) | (accumulator >> 33)) & _MASK64
    accumulator = (accumulator * _PRIME64_1) & _MASK64
    result = (_PRIME64_5 + 8) ^ accumulator
    result = ((result << 27) | (result >> 37)) & _MASK64
    result = (result * _PRIME64_1 + _PRIME64_4) & _MASK64
    return _avalanche(result)


def _xxh64_python(data: bytes) -> int:
    """Return XXH64(data, seed=0), the Parquet SBBF hash contract."""
    length = len(data)
    index = 0
    if length >= 32:
        v1 = (_PRIME64_1 + _PRIME64_2) & _MASK64
        v2 = _PRIME64_2
        v3 = 0
        v4 = (-_PRIME64_1) & _MASK64
        limit = length - 32
        while index <= limit:
            v1 = _round(v1, struct.unpack_from("<Q", data, index)[0])
            v2 = _round(v2, struct.unpack_from("<Q", data, index + 8)[0])
            v3 = _round(v3, struct.unpack_from("<Q", data, index + 16)[0])
            v4 = _round(v4, struct.unpack_from("<Q", data, index + 24)[0])
            index += 32
        result = (_rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)) & _MASK64
        result = _merge_round(result, v1)
        result = _merge_round(result, v2)
        result = _merge_round(result, v3)
        result = _merge_round(result, v4)
    else:
        result = _PRIME64_5

    result = (result + length) & _MASK64
    while index + 8 <= length:
        lane = _round(0, struct.unpack_from("<Q", data, index)[0])
        result ^= lane
        result = (_rotl64(result, 27) * _PRIME64_1 + _PRIME64_4) & _MASK64
        index += 8
    if index + 4 <= length:
        result ^= (struct.unpack_from("<I", data, index)[0] * _PRIME64_1) & _MASK64
        result = (_rotl64(result, 23) * _PRIME64_2 + _PRIME64_3) & _MASK64
        index += 4
    while index < length:
        result ^= (data[index] * _PRIME64_5) & _MASK64
        result = (_rotl64(result, 11) * _PRIME64_1) & _MASK64
        index += 1
    return _avalanche(result)


try:
    # Optional acceleration: pip install "pymilvus[bloom_filter]". The C implementation is the
    # same XXH64 the pure-Python fallback computes, so the blob is identical either way -- only
    # the build time changes. test_xxh64_python_matches_xxhash_c pins that equivalence whenever
    # the package is installed.
    from xxhash import xxh64_intdigest as _xxh64

    _PACK_INT64 = struct.Struct("<q").pack

    def _xxh64_int64(value: int) -> int:
        return _xxh64(_PACK_INT64(value))

except ImportError:  # pragma: no cover - depends on whether the extra is installed
    _xxh64 = _xxh64_python
    _xxh64_int64 = _xxh64_int64_python


# Members are hashed a chunk at a time so the temporary uint64 arrays stay cache-resident and
# peak memory does not scale with len(members). 16384 measured fastest; larger chunks fall off
# a cache cliff (256K was 1.7x slower and used 26 MiB more at 10M members).
_VECTOR_CHUNK = 1 << 14

# Pre-reduced so every numpy constant below is already in range -- computing them with numpy
# scalars would emit spurious overflow RuntimeWarnings.
_NP_P1 = np.uint64(_PRIME64_1)
_NP_P2 = np.uint64(_PRIME64_2)
_NP_P3 = np.uint64(_PRIME64_3)
_NP_P4 = np.uint64(_PRIME64_4)
_NP_SEED_LEN8 = np.uint64((_PRIME64_5 + 8) & _MASK64)
_NP_SALT = tuple(np.uint32(salt) for salt in _SALT)
_NP_32 = np.uint64(32)
_NP_MASK32 = np.uint64(0xFFFFFFFF)


def _int64_digests(lanes: np.ndarray) -> np.ndarray:
    """Vectorised XXH64 over each lane's 8-byte little-endian encoding.

    Mirrors :func:`_xxh64_int64_python` exactly, one numpy op per step.
    """
    accumulator = lanes.view(np.uint64) * _NP_P2
    accumulator = ((accumulator << np.uint64(31)) | (accumulator >> np.uint64(33))) * _NP_P1
    digest = _NP_SEED_LEN8 ^ accumulator
    digest = ((digest << np.uint64(27)) | (digest >> np.uint64(37))) * _NP_P1 + _NP_P4
    digest ^= digest >> np.uint64(33)
    digest *= _NP_P2
    digest ^= digest >> np.uint64(29)
    digest *= _NP_P3
    digest ^= digest >> _NP_32
    return digest


def _insert_digests(words: np.ndarray, digests: np.ndarray, num_blocks: int) -> None:
    """OR one chunk of XXH64 digests into the filter body.

    Split out from the hashing so both domains share it: the UTF-8 domain cannot vectorise its
    hash (XXH64's stripe and tail loops depend on each member's byte length) but its digests
    land in exactly the same bit positions, and doing this part per member in Python costs far
    more than the hashing does -- 10M strings hash in 0.4s with the C accelerator while the
    scalar insert loop they used to feed took ~10s.

    Two things make this faster than the obvious ``np.bitwise_or.at`` per word:

    * ``words`` is a uint64 view, so each 64-bit lane carries two of the block's eight 32-bit
      words and the scatter runs four times per member instead of eight. The view dtype pins
      little-endian, so lane = word[2i] | word[2i+1] << 32 holds on any host.
    * The scatter itself is a buffered ``words[idx] |= bits`` rather than the unbuffered
      ``ufunc.at``, which is an order of magnitude slower. A buffered scatter silently drops
      all but one write when indices repeat -- and they do repeat, since a 16384-member chunk
      landing in 524288 blocks collides a couple of hundred times -- but OR is idempotent and
      order-independent, so the writes that lost can simply be retried. Together: 2x.

    Retries are capped and the remainder handed to ``ufunc.at``. Without the cap a filter
    whose blocks are heavily oversubscribed (``BloomFilterBuilder(1, fpr)`` fed a million
    members, which the API allows) degenerates to one pass per colliding member and runs 11x
    *slower* than the unbuffered scatter it replaced.
    """
    blocks = np.uint64(num_blocks)
    lane_base = (((digests >> _NP_32) * blocks) >> _NP_32).astype(np.intp) * _LANES_PER_BLOCK
    key = (digests & _NP_MASK32).astype(np.uint32)
    masks = [np.uint64(1) << ((key * salt) >> np.uint32(27)).astype(np.uint64) for salt in _NP_SALT]

    for lane in range(_LANES_PER_BLOCK):
        index = lane_base + lane
        bits = masks[2 * lane] | (masks[2 * lane + 1] << _NP_32)
        for _ in range(_MAX_BUFFERED_PASSES):
            words[index] |= bits
            stale = (words[index] & bits) != bits
            if not stale.any():
                break
            index = index[stale]
            bits = bits[stale]
        else:
            np.bitwise_or.at(words, index, bits)


def _screen_int_members(chunk: Sequence[int]) -> None:
    """Reject anything np.fromiter would silently coerce.

    ``np.fromiter`` accepts where the scalar path raises: bool (an int subclass) becomes 0/1
    and float is truncated. ``map(type, ...)`` runs at C speed and the common all-int case
    settles in one set comparison.
    """
    member_types = set(map(type, chunk))
    if member_types != {int}:
        for member_type in member_types:
            if member_type is bool or not issubclass(member_type, int):
                raise ParamError(message="bloom filter members must be all int or all str")


def _screen_string_members(chunk: Sequence[str]) -> None:
    """Reject non-strings before calling their potentially string-like methods."""
    member_types = set(map(type, chunk))
    if member_types != {str}:
        for member_type in member_types:
            if not issubclass(member_type, str):
                raise ParamError(message="bloom filter members must be all int or all str")


def _chunks(values: Iterable[Any]) -> Iterator[Tuple[Iterable[Any], int]]:
    """Yield ``(chunk, size)`` pairs of at most ``_VECTOR_CHUNK`` members each.

    A list or tuple is already measurable, so its chunks are handed out as lazy ``islice``
    views and never copied -- materialising them into per-chunk lists costs ~4% on a 10M
    build for nothing. Anything else is a one-shot iterable (a generator, a DB cursor) whose
    length is unknown, so a chunk has to be pulled into a list before ``np.fromiter`` can be
    given a count. That bounded copy is exactly the trade the builder exists to make: one
    chunk resident instead of the whole set.

    The lazy chunks must be consumed before the next iteration, which every caller does.
    """
    if isinstance(values, (list, tuple)):
        iterator = iter(values)
        remaining = len(values)
        while remaining:
            size = _VECTOR_CHUNK if remaining > _VECTOR_CHUNK else remaining
            yield islice(iterator, size), size
            remaining -= size
        return
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, _VECTOR_CHUNK))
        if not chunk:
            return
        yield chunk, len(chunk)


def _add_int64_chunks(
    words: np.ndarray,
    values: Iterable[int],
    num_blocks: int,
    commit: Callable[[], None],
) -> None:
    """Hash and insert integer members a chunk at a time."""
    # A sequence can be screened in one C-speed pass; a one-shot iterable has to be screened
    # per chunk, since by then the earlier members are gone.
    prescreened = isinstance(values, (list, tuple))
    if prescreened:
        _screen_int_members(values)
    for chunk, size in _chunks(values):
        if not prescreened:
            _screen_int_members(chunk)
        try:
            lanes = np.fromiter(chunk, dtype=np.int64, count=size)
        except OverflowError as exc:
            raise ParamError(
                message="integer bloom filter members must fit in signed int64"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ParamError(message="bloom filter members must be all int or all str") from exc
        # The body cannot be rolled back without copying the whole filter. Mark the domain before
        # mutation so a failure during this or a later chunk leaves a valid partial commit.
        commit()
        _insert_digests(words, _int64_digests(lanes), num_blocks)


def _add_string_chunks(
    words: np.ndarray,
    values: Iterable[str],
    num_blocks: int,
    commit: Callable[[], None],
) -> None:
    """Hash and insert string members a chunk at a time."""
    prescreened = isinstance(values, (list, tuple))
    if prescreened:
        _screen_string_members(values)
    hash_bytes = _xxh64
    for chunk, size in _chunks(values):
        if not prescreened:
            _screen_string_members(chunk)
        digests = np.fromiter(
            (hash_bytes(value.encode("utf-8")) for value in chunk),
            dtype=np.uint64,
            count=size,
        )
        commit()
        _insert_digests(words, digests, num_blocks)


def _optimal_num_bytes(count: int, fpr: float) -> int:
    bits = -8.0 * count / math.log(1.0 - math.pow(fpr, 1.0 / 8.0))
    num_bits = int(bits) if 0 <= bits <= _MAX_FILTER_BYTES * 8 else _MAX_FILTER_BYTES * 8
    num_bits = max(num_bits, _MIN_FILTER_BYTES * 8)
    if num_bits & (num_bits - 1):
        num_bits = 1 << num_bits.bit_length()
    return min(num_bits, _MAX_FILTER_BYTES * 8) // 8


def _validate_fpr(fpr: float) -> float:
    if (
        not isinstance(fpr, (int, float))
        or not math.isfinite(fpr)
        or not _MIN_FPR <= fpr <= _MAX_FPR
    ):
        raise ParamError(message=f"bloom filter fpr must be in [{_MIN_FPR}, {_MAX_FPR}]")
    return float(fpr)


class BloomFilterBuilder:
    """Incrementally builds the blob :func:`build_bloom_filter` returns in one shot.

    Use this when the membership set does not comfortably fit in memory as a list. The filter
    is a fixed-size bit array sized from ``n`` and ``fpr`` alone -- never from the values --
    and insertion is order-independent and idempotent, so members can be streamed in from a
    cursor, a file or a paginated API and dropped as they go. Peak memory is then the filter
    itself (32 MiB for 24M members at the default fpr) rather than the filter plus the list;
    a 10M-element Python list of ints costs roughly 400 MB on its own.

    ``n`` only drives sizing. Adding more than ``n`` members is allowed and simply raises the
    effective false-positive rate above the target; adding fewer leaves the filter larger than
    it needed to be. A rough estimate is enough.

    Both add methods take an **iterable**, not one member at a time: they hash a chunk at a
    time with numpy, and a per-member Python call would cost more than the hashing does.
    Accumulate into batches of a few thousand and hand those over.

    Calls to the add methods and :meth:`build` are serialized per builder. Streaming adds are
    not transactional: once insertion of a chunk starts, that chunk remains committed if the
    iterable or a later chunk raises. Retrying members is safe because insertion is idempotent.

    .. code-block:: python

        builder = BloomFilterBuilder(10_000_000, fpr=0.005)
        for chunk in read_ids_in_chunks(conn, size=100_000):
            builder.add_int64_batch(chunk)
        blob = builder.build()

    Build the filter from the SAME value domain as the target field: integer fields hash
    int64, VARCHAR fields hash UTF-8. Mixing both domains in one filter is allowed here (a
    JSON path may legitimately hold both) but :func:`build_bloom_filter` rejects it.
    """

    def __init__(self, n: int, fpr: float = _DEFAULT_FPR) -> None:
        self._fpr = _validate_fpr(fpr)
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= _MASK64:
            raise ParamError(message="bloom filter member count must be a uint64")
        num_bytes = _optimal_num_bytes(n, self._fpr)
        self._num_blocks = num_bytes // 32
        self._n = n
        self._domains = 0
        self._lock = Lock()
        # The header is reserved in place so the blob is assembled in one buffer. Concatenating
        # a separate header would hold the body, its bytes() copy and the joined result live at
        # once, tripling peak memory for large member sets.
        self._buf = bytearray(_HEADER_SIZE + num_bytes)
        self._words = np.frombuffer(self._buf, dtype=_WORD64_DTYPE, offset=_HEADER_SIZE)

    def add_int64_batch(self, values: Iterable[int]) -> "BloomFilterBuilder":
        """Inserts integer members, hashed as their 8-byte little-endian encoding.

        Completed chunks remain committed if the iterable later raises.
        """
        with self._lock:
            _add_int64_chunks(
                self._words,
                values,
                self._num_blocks,
                lambda: self._commit_domain(_DOMAIN_INT64),
            )
        return self

    def add_string_batch(self, values: Iterable[str]) -> "BloomFilterBuilder":
        """Inserts string members, hashed as their raw UTF-8 bytes.

        Completed chunks remain committed if the iterable later raises.
        """
        with self._lock:
            _add_string_chunks(
                self._words,
                values,
                self._num_blocks,
                lambda: self._commit_domain(_DOMAIN_UTF8),
            )
        return self

    def _commit_domain(self, domain: int) -> None:
        self._domains |= domain

    @property
    def domains(self) -> int:
        """The value domains inserted so far. Zero means nothing was inserted."""
        return self._domains

    @property
    def num_blocks(self) -> int:
        """The number of 32-byte blocks in the filter body."""
        return self._num_blocks

    def build(self) -> bytes:
        """Stamps the MBF1 header and returns a copy of the envelope."""
        with self._lock:
            struct.pack_into(
                _HEADER_FORMAT,
                self._buf,
                0,
                b"MBF1",
                1,
                1,
                self._n,
                self._fpr,
                self._num_blocks,
                self._domains,
            )
            return bytes(self._buf)


def build_bloom_filter(members: Sequence[Union[int, str]], fpr: float = _DEFAULT_FPR) -> bytes:
    """Build an MBF1-wrapped Parquet Split-Block Bloom filter for bloom_match.

    Integer members target INT8/INT16/INT32/INT64 fields; string members target
    VARCHAR fields. The returned bytes are passed through ``filter_params``.

    Members must be homogeneous -- all int or all str. For a set too large to materialise as
    a list, stream it through :class:`BloomFilterBuilder` instead.
    """
    _validate_fpr(fpr)
    if not isinstance(members, (list, tuple)):
        raise ParamError(message="bloom filter members must be a list or tuple of int or str")

    builder = BloomFilterBuilder(len(members), fpr)
    if not members:
        pass
    elif isinstance(members[0], int) and not isinstance(members[0], bool):
        builder.add_int64_batch(members)
    elif isinstance(members[0], str):
        builder.add_string_batch(members)
    else:
        raise ParamError(message="bloom filter members must be all int or all str")

    return builder.build()


def _fill_scalar(
    buf: bytearray, members: Sequence[Union[int, str]], domain: int, num_blocks: int
) -> None:
    """Hash and insert members one at a time.

    Serves the UTF8 domain, which cannot be vectorised because XXH64's stripe and tail loops
    depend on each member's byte length. Also the reference the vectorised INT64 path is
    checked against by ``test_int64_vectorised_matches_scalar``.
    """
    int_domain = domain == _DOMAIN_INT64
    salt0, salt1, salt2, salt3, salt4, salt5, salt6, salt7 = _SALT
    unpack_block = _BLOCK_STRUCT.unpack_from
    pack_block = _BLOCK_STRUCT.pack_into
    hash_int64 = _xxh64_int64
    hash_bytes = _xxh64

    for value in members:
        if int_domain:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ParamError(message="bloom filter members must be all int or all str")
            if value < -(1 << 63) or value > (1 << 63) - 1:
                raise ParamError(message="integer bloom filter members must fit in signed int64")
            hash_value = hash_int64(value)
        else:
            if not isinstance(value, str):
                raise ParamError(message="bloom filter members must be all int or all str")
            hash_value = hash_bytes(value.encode("utf-8"))

        base = _HEADER_SIZE + ((((hash_value >> 32) * num_blocks) >> 32) << 5)
        key = hash_value & 0xFFFFFFFF
        word0, word1, word2, word3, word4, word5, word6, word7 = unpack_block(buf, base)
        pack_block(
            buf,
            base,
            word0 | 1 << ((key * salt0 & 0xFFFFFFFF) >> 27),
            word1 | 1 << ((key * salt1 & 0xFFFFFFFF) >> 27),
            word2 | 1 << ((key * salt2 & 0xFFFFFFFF) >> 27),
            word3 | 1 << ((key * salt3 & 0xFFFFFFFF) >> 27),
            word4 | 1 << ((key * salt4 & 0xFFFFFFFF) >> 27),
            word5 | 1 << ((key * salt5 & 0xFFFFFFFF) >> 27),
            word6 | 1 << ((key * salt6 & 0xFFFFFFFF) >> 27),
            word7 | 1 << ((key * salt7 & 0xFFFFFFFF) >> 27),
        )
