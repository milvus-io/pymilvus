import math
import struct
from itertools import islice
from typing import Sequence, Union

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
# The numpy view over those words must pin the same byte order: np.uint32 is host-native, which
# on a big-endian client would emit byte-swapped block words that a little-endian server reads
# as different bit positions.
_WORD_DTYPE = np.dtype("<u4")


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


def _fill_int64_vectorised(buf: bytearray, members: Sequence[int], num_blocks: int) -> None:
    """Hash and insert INT64 members with numpy, mirroring :func:`_xxh64_int64_python` exactly.

    ``np.fromiter`` performs the int64 range check in C, so out-of-range members surface as
    OverflowError instead of costing a per-member Python comparison.
    """
    # np.fromiter coerces silently where the scalar path raises: bool (an int subclass) becomes
    # 0/1 and float is truncated. Screen the distinct types up front -- map(type, ...) runs at C
    # speed, and the common all-int case settles in one set comparison.
    member_types = set(map(type, members))
    if member_types != {int}:
        for member_type in member_types:
            if member_type is bool or not issubclass(member_type, int):
                raise ParamError(message="bloom filter members must be all int or all str")

    words = np.frombuffer(buf, dtype=_WORD_DTYPE, offset=_HEADER_SIZE)
    blocks = np.uint64(num_blocks)
    values = iter(members)
    remaining = len(members)

    while remaining:
        size = _VECTOR_CHUNK if remaining > _VECTOR_CHUNK else remaining
        try:
            lanes = np.fromiter(islice(values, size), dtype=np.int64, count=size)
        except OverflowError as exc:
            raise ParamError(
                message="integer bloom filter members must fit in signed int64"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ParamError(message="bloom filter members must be all int or all str") from exc

        accumulator = lanes.view(np.uint64) * _NP_P2
        accumulator = ((accumulator << np.uint64(31)) | (accumulator >> np.uint64(33))) * _NP_P1
        digest = _NP_SEED_LEN8 ^ accumulator
        digest = ((digest << np.uint64(27)) | (digest >> np.uint64(37))) * _NP_P1 + _NP_P4
        digest ^= digest >> np.uint64(33)
        digest *= _NP_P2
        digest ^= digest >> np.uint64(29)
        digest *= _NP_P3
        digest ^= digest >> _NP_32

        word_base = (((digest >> _NP_32) * blocks) >> _NP_32).astype(np.intp) * 8
        key = (digest & _NP_MASK32).astype(np.uint32)
        for index, salt in enumerate(_NP_SALT):
            bit = np.uint32(1) << ((key * salt) >> np.uint32(27))
            # Unbuffered: several members routinely land in the same block, so the OR has to
            # accumulate rather than let one write win.
            np.bitwise_or.at(words, word_base + index, bit)

        remaining -= size


def _optimal_num_bytes(count: int, fpr: float) -> int:
    bits = -8.0 * count / math.log(1.0 - math.pow(fpr, 1.0 / 8.0))
    num_bits = int(bits) if 0 <= bits <= _MAX_FILTER_BYTES * 8 else _MAX_FILTER_BYTES * 8
    num_bits = max(num_bits, _MIN_FILTER_BYTES * 8)
    if num_bits & (num_bits - 1):
        num_bits = 1 << num_bits.bit_length()
    return min(num_bits, _MAX_FILTER_BYTES * 8) // 8


def build_bloom_filter(members: Sequence[Union[int, str]], fpr: float = _DEFAULT_FPR) -> bytes:
    """Build an MBF1-wrapped Parquet Split-Block Bloom filter for bloom_match.

    Integer members target INT8/INT16/INT32/INT64 fields; string members target
    VARCHAR fields. The returned bytes are passed through ``filter_params``.
    """
    if (
        not isinstance(fpr, (int, float))
        or not math.isfinite(fpr)
        or not _MIN_FPR <= fpr <= _MAX_FPR
    ):
        raise ParamError(message=f"bloom filter fpr must be in [{_MIN_FPR}, {_MAX_FPR}]")
    if not isinstance(members, (list, tuple)):
        raise ParamError(message="bloom filter members must be a list or tuple of int or str")
    if not members:
        domain = 0
    elif isinstance(members[0], int) and not isinstance(members[0], bool):
        domain = _DOMAIN_INT64
    elif isinstance(members[0], str):
        domain = _DOMAIN_UTF8
    else:
        raise ParamError(message="bloom filter members must be all int or all str")

    count = len(members)
    num_bytes = _optimal_num_bytes(count, float(fpr))
    num_blocks = num_bytes // 32
    # Reserve the header in place so the blob is assembled in one buffer. Concatenating a separate
    # header would hold the body, its bytes() copy and the joined result live at once, tripling
    # peak memory for large member sets.
    buf = bytearray(_HEADER_SIZE + num_bytes)
    struct.pack_into(_HEADER_FORMAT, buf, 0, b"MBF1", 1, 1, count, float(fpr), num_blocks, domain)

    if domain == _DOMAIN_INT64:
        _fill_int64_vectorised(buf, members, num_blocks)
    elif domain == _DOMAIN_UTF8:
        _fill_scalar(buf, members, domain, num_blocks)

    return bytes(buf)


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


def _fill_scalar_int64(buf: bytearray, members: Sequence[int], num_blocks: int) -> None:
    """Scalar INT64 insertion, signature-compatible with :func:`_fill_int64_vectorised`.

    Not on the default path -- it exists so the vectorised implementation has a reference to be
    checked against, and as a drop-in should numpy ever need to be bypassed.
    """
    _fill_scalar(buf, members, _DOMAIN_INT64, num_blocks)
