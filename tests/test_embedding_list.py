"""Tests for EmbeddingList with non-float vector types and placeholder type selection."""

import numpy as np
import pytest
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.prepare import Prepare
from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.grpc_gen import common_pb2


class TestEmbeddingListBytesInput:
    """Test EmbeddingList.add() handles bytes input correctly."""

    def test_add_bytes_with_uint8_dtype(self):
        """bytes input should be converted to uint8 ndarray when dtype is uint8."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        raw = b"\x01\x02\x03\x04"
        el.add(raw)
        assert len(el) == 1
        assert el[0].dtype == np.uint8
        np.testing.assert_array_equal(el[0], np.array([1, 2, 3, 4], dtype=np.uint8))

    def test_add_bytes_without_dtype_defaults_to_uint8(self):
        """bytes input without explicit dtype should default to uint8."""
        el = EmbeddingList()
        el.add(b"\xff\x00\xab")
        assert el[0].dtype == np.uint8
        assert len(el[0]) == 3

    def test_add_bytes_with_int8_dtype(self):
        """bytes input with int8 dtype should produce int8 ndarray."""
        el = EmbeddingList(dtype=DataType.INT8_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        assert el[0].dtype == np.int8

    def test_add_multiple_bytes(self):
        """Multiple bytes embeddings should work and maintain dimension consistency."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        el.add(b"\x05\x06\x07\x08")
        assert len(el) == 2
        assert el.dim == 4

    def test_add_bytes_dimension_mismatch_raises(self):
        """Adding bytes with different length should raise ValueError."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        with pytest.raises(ValueError, match="dimension"):
            el.add(b"\x01\x02")

    def test_to_flat_array_with_bytes_input(self):
        """to_flat_array should return concatenated uint8 ndarray."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02")
        el.add(b"\x03\x04")
        flat = el.to_flat_array()
        assert flat.dtype == np.uint8
        np.testing.assert_array_equal(flat, np.array([1, 2, 3, 4], dtype=np.uint8))


class TestEmbeddingListBFloat16Dtype:
    """Test EmbeddingList._parse_dtype() handles bfloat16 correctly."""

    def test_parse_dtype_bfloat16_not_float16(self):
        """DataType.BFLOAT16_VECTOR should not produce float16 when bfloat16 is available."""
        el = EmbeddingList(dtype=DataType.BFLOAT16_VECTOR)
        try:
            expected = np.dtype("bfloat16")
        except TypeError:
            # bfloat16 not available, fallback to float16 is acceptable
            expected = np.dtype(np.float16)
        assert el._dtype == expected

    def test_infer_dtype_preserves_bfloat16(self):
        """When adding bfloat16 arrays, the dtype should be preserved, not converted."""
        try:
            arr = np.array([1.0, 2.0], dtype="bfloat16")
        except TypeError:
            pytest.skip("bfloat16 not available in this numpy version")
        el = EmbeddingList()
        el.add(arr)
        assert el.dtype == np.dtype("bfloat16")


class TestPreparePlaceholderBinaryEmbeddingList:
    """Test _prepare_placeholder_str uses EmbListBinaryVector for binary EmbeddingList."""

    def _deserialize_placeholder(self, serialized_bytes):
        """Helper to deserialize placeholder group and return the PlaceholderValue."""
        pg = common_pb2.PlaceholderGroup()
        pg.ParseFromString(serialized_bytes)
        return pg.placeholders[0]

    def test_uint8_ndarray_embedding_list_uses_emb_list_binary(self):
        """uint8 ndarray data with is_embedding_list=True should use EmbListBinaryVector."""
        data = [np.array([1, 2, 3, 4], dtype=np.uint8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListBinaryVector

    def test_uint8_ndarray_no_embedding_list_uses_binary(self):
        """uint8 ndarray data without is_embedding_list should use BinaryVector."""
        data = [np.array([1, 2, 3, 4], dtype=np.uint8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=False)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.BinaryVector

    def test_bytes_data_embedding_list_uses_emb_list_binary(self):
        """bytes data with is_embedding_list=True should use EmbListBinaryVector."""
        data = [b"\x01\x02\x03\x04"]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListBinaryVector

    def test_bytes_data_no_embedding_list_uses_binary(self):
        """bytes data without is_embedding_list should use BinaryVector."""
        data = [b"\x01\x02\x03\x04"]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=False)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.BinaryVector

    def test_int8_embedding_list_uses_emb_list_int8(self):
        """int8 ndarray with is_embedding_list=True should use EmbListInt8Vector."""
        data = [np.array([1, -1, 2, -2], dtype=np.int8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListInt8Vector

    def test_float16_embedding_list_uses_emb_list_float16(self):
        """float16 ndarray with is_embedding_list=True should use EmbListFloat16Vector."""
        data = [np.array([1.0, 2.0], dtype=np.float16)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListFloat16Vector


class TestEmbeddingListEndToEnd:
    """End-to-end test: EmbeddingList creation -> to_flat_array -> placeholder serialization."""

    def _deserialize_placeholder(self, serialized_bytes):
        pg = common_pb2.PlaceholderGroup()
        pg.ParseFromString(serialized_bytes)
        return pg.placeholders[0]

    def test_binary_embedding_list_full_flow(self):
        """Binary EmbeddingList should produce EmbListBinaryVector placeholder."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        el.add(b"\x05\x06\x07\x08")

        flat = el.to_flat_array()
        assert flat.dtype == np.uint8

        data = [flat]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListBinaryVector
        assert pv.values[0] == flat.tobytes()

    def test_int8_embedding_list_full_flow(self):
        """Int8 EmbeddingList should produce EmbListInt8Vector placeholder."""
        el = EmbeddingList(dtype=DataType.INT8_VECTOR)
        el.add(np.array([1, -1, 2, -2], dtype=np.int8))
        el.add(np.array([3, -3, 4, -4], dtype=np.int8))

        flat = el.to_flat_array()
        assert flat.dtype == np.int8

        data = [flat]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListInt8Vector

    def test_float16_embedding_list_full_flow(self):
        """Float16 EmbeddingList should produce EmbListFloat16Vector placeholder."""
        el = EmbeddingList(dtype=DataType.FLOAT16_VECTOR)
        el.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16))

        flat = el.to_flat_array()
        assert flat.dtype == np.float16

        data = [flat]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = self._deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListFloat16Vector
