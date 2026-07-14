import math
import struct
from typing import Sequence, Union

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


def _rotl64(value: int, count: int) -> int:
    return ((value << count) | (value >> (64 - count))) & _MASK64


def _round(accumulator: int, value: int) -> int:
    accumulator = (accumulator + value * _PRIME64_2) & _MASK64
    accumulator = _rotl64(accumulator, 31)
    return (accumulator * _PRIME64_1) & _MASK64


def _merge_round(accumulator: int, value: int) -> int:
    accumulator ^= _round(0, value)
    return (accumulator * _PRIME64_1 + _PRIME64_4) & _MASK64


def _xxh64(data: bytes) -> int:
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
    result ^= result >> 33
    result = (result * _PRIME64_2) & _MASK64
    result ^= result >> 29
    result = (result * _PRIME64_3) & _MASK64
    return result ^ (result >> 32)


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
    if members and all(isinstance(value, int) and not isinstance(value, bool) for value in members):
        encoded = []
        for value in members:
            if value < -(1 << 63) or value > (1 << 63) - 1:
                raise ParamError(message="integer bloom filter members must fit in signed int64")
            encoded.append(struct.pack("<q", value))
    elif members and all(isinstance(value, str) for value in members):
        encoded = [value.encode("utf-8") for value in members]
    elif not members:
        encoded = []
    else:
        raise ParamError(message="bloom filter members must be all int or all str")

    num_bytes = _optimal_num_bytes(len(encoded), float(fpr))
    num_blocks = num_bytes // 32
    body = bytearray(num_bytes)
    for value in encoded:
        hash_value = _xxh64(value)
        block = ((hash_value >> 32) * num_blocks) >> 32
        key = hash_value & 0xFFFFFFFF
        base = block * 32
        for index, salt in enumerate(_SALT):
            offset = base + index * 4
            word = struct.unpack_from("<I", body, offset)[0]
            word |= 1 << ((key * salt & 0xFFFFFFFF) >> 27)
            struct.pack_into("<I", body, offset, word)

    header = struct.pack("<4sHHQdII", b"MBF1", 1, 1, len(encoded), float(fpr), num_blocks, 0)
    return header + bytes(body)
