"""Tests for EmbeddingList with non-float vector types and placeholder type selection."""

import numpy as np
import pytest
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.prepare import Prepare
from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2


def _deserialize_placeholder(serialized_bytes):
    """Helper to deserialize placeholder group and return the PlaceholderValue."""
    pg = common_pb2.PlaceholderGroup()
    pg.ParseFromString(serialized_bytes)
    return pg.placeholders[0]


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

    def test_uint8_ndarray_embedding_list_uses_emb_list_binary(self):
        """uint8 ndarray data with is_embedding_list=True should use EmbListBinaryVector."""
        data = [np.array([1, 2, 3, 4], dtype=np.uint8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListBinaryVector

    def test_uint8_ndarray_no_embedding_list_uses_binary(self):
        """uint8 ndarray data without is_embedding_list should use BinaryVector."""
        data = [np.array([1, 2, 3, 4], dtype=np.uint8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=False)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.BinaryVector

    def test_bytes_data_embedding_list_uses_emb_list_binary(self):
        """bytes data with is_embedding_list=True should use EmbListBinaryVector."""
        data = [b"\x01\x02\x03\x04"]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListBinaryVector

    def test_bytes_data_no_embedding_list_uses_binary(self):
        """bytes data without is_embedding_list should use BinaryVector."""
        data = [b"\x01\x02\x03\x04"]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=False)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.BinaryVector

    def test_int8_embedding_list_uses_emb_list_int8(self):
        """int8 ndarray with is_embedding_list=True should use EmbListInt8Vector."""
        data = [np.array([1, -1, 2, -2], dtype=np.int8)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListInt8Vector

    def test_float16_embedding_list_uses_emb_list_float16(self):
        """float16 ndarray with is_embedding_list=True should use EmbListFloat16Vector."""
        data = [np.array([1.0, 2.0], dtype=np.float16)]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListFloat16Vector


class TestEmbeddingListEndToEnd:
    """End-to-end test: EmbeddingList creation -> to_flat_array -> placeholder serialization."""

    def test_binary_embedding_list_full_flow(self):
        """Binary EmbeddingList should produce EmbListBinaryVector placeholder."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        el.add(b"\x05\x06\x07\x08")

        flat = el.to_flat_array()
        assert flat.dtype == np.uint8

        data = [flat]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
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
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListInt8Vector

    def test_float16_embedding_list_full_flow(self):
        """Float16 EmbeddingList should produce EmbListFloat16Vector placeholder."""
        el = EmbeddingList(dtype=DataType.FLOAT16_VECTOR)
        el.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16))

        flat = el.to_flat_array()
        assert flat.dtype == np.float16

        data = [flat]
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == PlaceholderType.EmbListFloat16Vector


class TestEmbeddingListInit:
    """Tests for __init__ branches: ndarray 1D/2D/3D, list, wrong type."""

    def test_init_1d_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        el = EmbeddingList(arr)
        assert len(el) == 1
        np.testing.assert_array_equal(el[0], arr)

    def test_init_2d_ndarray(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        el = EmbeddingList(arr)
        assert len(el) == 2

    def test_init_3d_ndarray_raises(self):
        arr = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="1D or 2D"):
            EmbeddingList(arr)

    def test_init_list_of_arrays(self):
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        el = EmbeddingList(vecs)
        assert len(el) == 2

    def test_init_wrong_type_raises(self):
        with pytest.raises(TypeError, match="numpy array or list"):
            EmbeddingList(42)

    def test_init_none_is_empty(self):
        el = EmbeddingList(None)
        assert el.is_empty


class TestEmbeddingListParseDtype:
    """Tests for _parse_dtype branches."""

    def test_parse_numpy_dtype(self):
        el = EmbeddingList(dtype=np.dtype(np.float32))
        assert el._dtype == np.dtype(np.float32)

    def test_parse_string_dtype(self):
        el = EmbeddingList(dtype="float32")
        assert el._dtype == np.dtype(np.float32)

    def test_parse_datatype_float_vector(self):
        el = EmbeddingList(dtype=DataType.FLOAT_VECTOR)
        assert el._dtype == np.dtype(np.float32)

    def test_parse_datatype_float16_vector(self):
        el = EmbeddingList(dtype=DataType.FLOAT16_VECTOR)
        assert el._dtype == np.dtype(np.float16)

    def test_parse_datatype_int8_vector(self):
        el = EmbeddingList(dtype=DataType.INT8_VECTOR)
        assert el._dtype == np.dtype(np.int8)

    def test_parse_unsupported_datatype_raises(self):
        with pytest.raises(ParamError):
            EmbeddingList(dtype=DataType.INT64)

    def test_parse_invalid_type_raises(self):
        with pytest.raises(TypeError, match="dtype must be"):
            EmbeddingList(dtype=12345)

    def test_parse_bfloat16_fallback(self):
        """When bfloat16 is unavailable, should fall back to float16."""
        el = EmbeddingList(dtype=DataType.BFLOAT16_VECTOR)
        try:
            expected = np.dtype("bfloat16")
        except TypeError:
            expected = np.dtype(np.float16)
        assert el._dtype == expected


class TestEmbeddingListAdd:
    """Tests for add() validation branches."""

    def test_add_non_1d_raises(self):
        el = EmbeddingList()
        with pytest.raises(ValueError, match="1D"):
            el.add(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_add_wrong_dim_vs_constructor_dim_raises(self):
        el = EmbeddingList(dim=4)
        with pytest.raises(ValueError, match="dimension"):
            el.add(np.array([1.0, 2.0]))

    def test_add_wrong_dim_vs_existing_raises(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="dimension"):
            el.add(np.array([1.0, 2.0, 3.0]))

    def test_add_list_converts(self):
        el = EmbeddingList()
        el.add([1.0, 2.0, 3.0])
        assert len(el) == 1

    def test_add_dtype_conversion(self):
        el = EmbeddingList(dtype=DataType.FLOAT_VECTOR)
        el.add(np.array([1.0, 2.0], dtype=np.float64))
        assert el[0].dtype == np.float32

    def test_add_float64_infers_float32(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0], dtype=np.float64))
        assert el.dtype == np.dtype(np.float32)

    def test_add_chaining(self):
        el = EmbeddingList()
        result = el.add(np.array([1.0, 2.0])).add(np.array([3.0, 4.0]))
        assert result is el
        assert len(el) == 2


class TestEmbeddingListAddBatch:
    """Tests for add_batch()."""

    def test_add_batch_2d_array(self):
        el = EmbeddingList()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        el.add_batch(arr)
        assert len(el) == 2

    def test_add_batch_non_2d_raises(self):
        el = EmbeddingList()
        with pytest.raises(ValueError, match="2D"):
            el.add_batch(np.array([1.0, 2.0]))

    def test_add_batch_list(self):
        el = EmbeddingList()
        el.add_batch([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        assert len(el) == 2

    def test_add_batch_chaining(self):
        el = EmbeddingList()
        result = el.add_batch(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        assert result is el


@pytest.mark.parametrize(
    "count,dim,dtype,seed,expected_dtype",
    [
        (3, 4, None, 0, np.dtype(np.float32)),
        (2, 8, DataType.BINARY_VECTOR, 1, np.dtype(np.uint8)),
        (2, 4, DataType.INT8_VECTOR, 2, np.dtype(np.int8)),
        (2, 4, DataType.FLOAT16_VECTOR, 3, np.dtype(np.float16)),
        (2, 4, "float64", 4, np.dtype(np.float64)),
        (1, 4, None, 0, np.dtype(np.float32)),  # None dtype defaults to float32
    ],
)
def test_from_random(count, dim, dtype, seed, expected_dtype):
    """Test _from_random_test classmethod with various dtypes."""
    kwargs = {"seed": seed}
    if dtype is not None:
        kwargs["dtype"] = dtype
    el = EmbeddingList._from_random_test(count, dim, **kwargs)
    assert el.dtype == expected_dtype


def test_from_random_unsupported_dtype_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        EmbeddingList._from_random_test(2, 4, dtype="complex64")


class TestEmbeddingListFromRandom:
    """Tests for _from_random_test classmethod - kept for count/dim assertions."""

    def test_from_random_float32_count_dim(self):
        el = EmbeddingList._from_random_test(3, 4, seed=0)
        assert len(el) == 3
        assert el.dim == 4

    def test_from_random_uint8_count(self):
        el = EmbeddingList._from_random_test(2, 8, dtype=DataType.BINARY_VECTOR, seed=1)
        assert len(el) == 2


class TestEmbeddingListConversions:
    """Tests for to_flat_array, to_numpy, clear."""

    def test_to_flat_array_empty_raises(self):
        el = EmbeddingList()
        with pytest.raises(ValueError, match="empty"):
            el.to_flat_array()

    def test_to_numpy_basic(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0], dtype=np.float32))
        el.add(np.array([3.0, 4.0], dtype=np.float32))
        arr = el.to_numpy()
        assert arr.shape == (2, 2)

    def test_to_numpy_empty_raises(self):
        el = EmbeddingList()
        with pytest.raises(ValueError, match="empty"):
            el.to_numpy()

    def test_clear(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        result = el.clear()
        assert result is el
        assert len(el) == 0


class TestEmbeddingListProperties:
    """Tests for all properties and magic methods."""

    def test_iter(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        el.add(np.array([3.0, 4.0]))
        items = list(el)
        assert len(items) == 2

    def test_dim_empty_no_hint(self):
        el = EmbeddingList()
        assert el.dim == 0

    def test_dim_empty_with_hint(self):
        el = EmbeddingList(dim=8)
        assert el.dim == 8

    def test_shape(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        el.add(np.array([3.0, 4.0]))
        assert el.shape == (2, 2)

    def test_total_dim(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        el.add(np.array([3.0, 4.0]))
        assert el.total_dim == 4

    def test_is_empty_true(self):
        el = EmbeddingList()
        assert el.is_empty is True

    def test_is_empty_false(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        assert el.is_empty is False

    def test_nbytes_empty(self):
        el = EmbeddingList()
        assert el.nbytes == 0

    def test_nbytes_non_empty(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0], dtype=np.float32))
        assert el.nbytes == 8

    def test_repr_with_dtype(self):
        el = EmbeddingList(dtype=DataType.FLOAT_VECTOR)
        el.add(np.array([1.0, 2.0]))
        r = repr(el)
        assert "EmbeddingList" in r
        assert "count=1" in r

    def test_repr_no_dtype(self):
        el = EmbeddingList()
        r = repr(el)
        assert "count=0" in r

    def test_str_empty(self):
        el = EmbeddingList()
        assert str(el) == "EmbeddingList(empty)"

    def test_str_non_empty(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        s = str(el)
        assert "1 embeddings" in s
        assert "dimension 2" in s
