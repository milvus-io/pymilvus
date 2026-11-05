"""Tests for EmbeddingList with non-float vector types and placeholder type selection."""

from unittest.mock import patch

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
    assert len(pg.placeholders) == 1
    return pg.placeholders[0]


class TestEmbeddingListBytesInput:
    """Test EmbeddingList.add() handles bytes input correctly."""

    @pytest.mark.parametrize(
        "dtype,raw,expected_dtype,expected",
        [
            pytest.param(
                DataType.BINARY_VECTOR, b"\x01\x02\x03\x04", np.uint8, [1, 2, 3, 4], id="binary"
            ),
            pytest.param(None, b"\xff\x00\xab", np.uint8, [255, 0, 171], id="inferred-uint8"),
            pytest.param(
                DataType.INT8_VECTOR,
                b"\x01\x7f\x80\xff",
                np.int8,
                [1, 127, -128, -1],
                id="signed-int8",
            ),
        ],
    )
    def test_add_bytes(self, dtype, raw, expected_dtype, expected):
        el = EmbeddingList(dtype=dtype)
        el.add(raw)

        assert len(el) == 1
        assert el.dtype == expected_dtype
        assert el[0].dtype == expected_dtype
        assert el[0].tolist() == expected

    def test_add_bytes_dimension_mismatch_raises(self):
        """Adding bytes with different length should raise ValueError."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02\x03\x04")
        with pytest.raises(ValueError, match="dimension"):
            el.add(b"\x01\x02")
        assert len(el) == 1
        assert el.dim == 4
        np.testing.assert_array_equal(el.to_flat_array(), [1, 2, 3, 4])

    def test_to_flat_array_with_bytes_input(self):
        """to_flat_array should return concatenated uint8 ndarray."""
        el = EmbeddingList(dtype=DataType.BINARY_VECTOR)
        el.add(b"\x01\x02")
        el.add(b"\x03\x04")
        flat = el.to_flat_array()
        assert len(el) == 2
        assert el.dim == 2
        assert el[0].tolist() == [1, 2]
        assert el[1].tolist() == [3, 4]
        assert flat.dtype == np.uint8
        np.testing.assert_array_equal(flat, np.array([1, 2, 3, 4], dtype=np.uint8))


class TestEmbeddingListBFloat16Dtype:
    """Test native bfloat16 values and the float16 compatibility fallback."""

    def test_parse_dtype_bfloat16_not_float16(self):
        """DataType.BFLOAT16_VECTOR should not produce float16 when bfloat16 is available."""
        pytest.importorskip("ml_dtypes")
        el = EmbeddingList(dtype=DataType.BFLOAT16_VECTOR)
        expected = np.dtype("bfloat16")
        assert el.dtype == expected
        el.add([1.5, 2.5])
        assert el.to_flat_array().dtype == expected
        assert el.to_flat_array().tolist() == [1.5, 2.5]

    def test_infer_dtype_preserves_bfloat16(self):
        """When adding bfloat16 arrays, the dtype should be preserved, not converted."""
        pytest.importorskip("ml_dtypes")
        arr = np.array([1.0, 2.0], dtype="bfloat16")
        el = EmbeddingList()
        el.add(arr)
        assert el.dtype == np.dtype("bfloat16")
        assert el.to_flat_array().dtype == np.dtype("bfloat16")
        np.testing.assert_array_equal(el.to_flat_array(), [1.0, 2.0])


class TestPreparePlaceholderEmbeddingList:
    """Placeholder types and payloads must preserve each query in order."""

    @pytest.mark.parametrize("is_embedding_list", [False, True], ids=["vector", "embedding-list"])
    @pytest.mark.parametrize(
        "data,vector_type,embedding_list_type,expected_values",
        [
            pytest.param(
                [np.array([1, 2, 3, 4], dtype=np.uint8), np.array([5, 6, 7, 8], dtype=np.uint8)],
                PlaceholderType.BinaryVector,
                PlaceholderType.EmbListBinaryVector,
                [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"],
                id="uint8",
            ),
            pytest.param(
                [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"],
                PlaceholderType.BinaryVector,
                PlaceholderType.EmbListBinaryVector,
                [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"],
                id="bytes",
            ),
            pytest.param(
                [np.array([1, -1, 2, -2], dtype=np.int8), np.array([3, -3, 4, -4], dtype=np.int8)],
                PlaceholderType.Int8Vector,
                PlaceholderType.EmbListInt8Vector,
                [b"\x01\xff\x02\xfe", b"\x03\xfd\x04\xfc"],
                id="int8",
            ),
            pytest.param(
                [np.array([1.0, 2.0], dtype=np.float16), np.array([3.0, 4.0], dtype=np.float16)],
                PlaceholderType.FLOAT16_VECTOR,
                PlaceholderType.EmbListFloat16Vector,
                [b"\x00\x3c\x00\x40", b"\x00\x42\x00\x44"],
                id="float16",
            ),
        ],
    )
    def test_placeholder_type_and_values(
        self, data, vector_type, embedding_list_type, expected_values, is_embedding_list
    ):
        result = Prepare._prepare_placeholder_str(data, is_embedding_list=is_embedding_list)
        pv = _deserialize_placeholder(result)

        assert pv.tag == "$0"
        assert pv.type == (embedding_list_type if is_embedding_list else vector_type)
        assert list(pv.values) == expected_values


class TestEmbeddingListEndToEnd:
    """End-to-end test: EmbeddingList creation -> to_flat_array -> placeholder serialization."""

    @pytest.mark.parametrize(
        "dtype,embeddings,expected_dtype,expected_values,placeholder_type,expected_bytes",
        [
            pytest.param(
                DataType.BINARY_VECTOR,
                [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"],
                np.uint8,
                [1, 2, 3, 4, 5, 6, 7, 8],
                PlaceholderType.EmbListBinaryVector,
                b"\x01\x02\x03\x04\x05\x06\x07\x08",
                id="binary",
            ),
            pytest.param(
                DataType.INT8_VECTOR,
                [np.array([1, -1, 2, -2], dtype=np.int8), np.array([3, -3, 4, -4], dtype=np.int8)],
                np.int8,
                [1, -1, 2, -2, 3, -3, 4, -4],
                PlaceholderType.EmbListInt8Vector,
                b"\x01\xff\x02\xfe\x03\xfd\x04\xfc",
                id="int8",
            ),
            pytest.param(
                DataType.FLOAT16_VECTOR,
                [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)],
                np.float16,
                [1.0, 2.0, 3.0, 4.0],
                PlaceholderType.EmbListFloat16Vector,
                b"\x00\x3c\x00\x40\x00\x42\x00\x44",
                id="float16",
            ),
        ],
    )
    def test_embedding_list_full_flow(
        self, dtype, embeddings, expected_dtype, expected_values, placeholder_type, expected_bytes
    ):
        el = EmbeddingList(dtype=dtype)
        for embedding in embeddings:
            el.add(embedding)
        flat = el.to_flat_array()
        assert flat.dtype == expected_dtype
        np.testing.assert_array_equal(flat, expected_values)

        result = Prepare._prepare_placeholder_str([flat], is_embedding_list=True)
        pv = _deserialize_placeholder(result)
        assert pv.type == placeholder_type
        assert list(pv.values) == [expected_bytes]


class TestEmbeddingListInit:
    """Tests for __init__ branches: ndarray 1D/2D/3D, list, wrong type."""

    @pytest.mark.parametrize(
        "embeddings,expected",
        [
            pytest.param(
                np.array([1.0, 2.0, 3.0], dtype=np.float32), [[1.0, 2.0, 3.0]], id="1d-array"
            ),
            pytest.param(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                [[1.0, 2.0], [3.0, 4.0]],
                id="2d-array",
            ),
            pytest.param(
                [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                [[1.0, 2.0], [3.0, 4.0]],
                id="list-of-arrays",
            ),
        ],
    )
    def test_init_preserves_embeddings(self, embeddings, expected):
        el = EmbeddingList(embeddings)
        assert len(el) == len(expected)
        assert el.dim == len(expected[0])
        assert el.to_numpy().dtype == np.float32
        np.testing.assert_array_equal(el.to_numpy(), expected)
        np.testing.assert_array_equal(el[0], expected[0])

    def test_init_3d_ndarray_raises(self):
        arr = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="1D or 2D"):
            EmbeddingList(arr)

    def test_init_wrong_type_raises(self):
        with pytest.raises(TypeError, match="numpy array or list"):
            EmbeddingList(42)

    def test_init_none_is_empty(self):
        el = EmbeddingList(None)
        assert el.is_empty


class TestEmbeddingListParseDtype:
    """Requested dtypes control the stored and serialized values."""

    @pytest.mark.parametrize(
        "dtype,expected_dtype",
        [
            pytest.param(np.dtype(np.float32), np.float32, id="numpy-dtype"),
            pytest.param("float32", np.float32, id="dtype-name"),
            pytest.param(DataType.FLOAT_VECTOR, np.float32, id="float32"),
            pytest.param(DataType.FLOAT16_VECTOR, np.float16, id="float16"),
            pytest.param(DataType.INT8_VECTOR, np.int8, id="int8"),
            pytest.param(DataType.BINARY_VECTOR, np.uint8, id="binary"),
        ],
    )
    def test_requested_dtype_controls_values(self, dtype, expected_dtype):
        el = EmbeddingList(dtype=dtype)
        assert el.dtype == expected_dtype

        el.add([1, 2])
        flat = el.to_flat_array()
        assert flat.dtype == expected_dtype
        assert flat.tolist() == [1, 2]

    def test_parse_unsupported_datatype_raises(self):
        with pytest.raises(ParamError):
            EmbeddingList(dtype=DataType.INT64)

    def test_parse_invalid_type_raises(self):
        with pytest.raises(TypeError, match="dtype must be"):
            EmbeddingList(dtype=12345)

    def test_bfloat16_falls_back_when_dtype_is_unavailable(self):
        with patch(
            "pymilvus.client.embedding_list.require_numpy_dtype",
            return_value="missing_bfloat16",
        ):
            el = EmbeddingList(dtype=DataType.BFLOAT16_VECTOR)

        el.add([1.5, 2.5])
        assert el.dtype == np.dtype(np.float16)
        assert el.to_flat_array().dtype == np.dtype(np.float16)
        assert el.to_flat_array().tolist() == [1.5, 2.5]

    def test_parse_datatype_missing_registry_dtype_raises_param_error(self):
        with patch(
            "pymilvus.client.embedding_list.require_numpy_dtype",
            side_effect=ParamError(message="Unsupported DataType: 101"),
        ):
            with pytest.raises(ParamError):
                EmbeddingList(dtype=DataType.FLOAT_VECTOR)

    def test_parse_datatype_missing_registry_fallback_raises_type_error(self):
        with patch(
            "pymilvus.client.embedding_list.require_numpy_dtype", return_value="missing_float"
        ):
            with patch(
                "pymilvus.client.embedding_list.get_numpy_fallback_dtype", return_value=None
            ):
                with pytest.raises(TypeError):
                    EmbeddingList(dtype=DataType.FLOAT_VECTOR)

    def test_parse_datatype_invalid_registry_fallback_raises_type_error(self):
        with patch(
            "pymilvus.client.embedding_list.require_numpy_dtype",
            return_value="missing_bfloat16",
        ):
            with patch(
                "pymilvus.client.embedding_list.get_numpy_fallback_dtype",
                return_value="missing_fallback",
            ):
                with pytest.raises(TypeError):
                    EmbeddingList(dtype=DataType.BFLOAT16_VECTOR)


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
        assert el.shape == (0, 4)
        assert el.dtype is None

    def test_add_wrong_dim_vs_existing_raises(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="dimension"):
            el.add(np.array([1.0, 2.0, 3.0]))
        assert el.shape == (1, 2)
        np.testing.assert_array_equal(el.to_numpy(), [[1.0, 2.0]])

    def test_add_list_converts(self):
        el = EmbeddingList()
        el.add([1.0, 2.0, 3.0])
        assert len(el) == 1
        assert el[0].dtype == np.float32
        np.testing.assert_array_equal(el[0], [1.0, 2.0, 3.0])

    def test_add_dtype_conversion(self):
        el = EmbeddingList(dtype=DataType.FLOAT_VECTOR)
        el.add(np.array([1.0, 2.0], dtype=np.float64))
        assert el[0].dtype == np.float32
        np.testing.assert_array_equal(el[0], [1.0, 2.0])

    def test_add_float64_infers_float32(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0], dtype=np.float64))
        assert el.dtype == np.dtype(np.float32)
        assert el[0].dtype == np.float32
        np.testing.assert_array_equal(el[0], [1.0, 2.0])

    def test_later_vectors_use_the_first_vectors_dtype(self):
        el = EmbeddingList(np.array([1.5, -2.5], dtype=np.float16))
        el.add(np.array([3.5, -4.5], dtype=np.float64))

        assert el.dtype == np.float16
        assert el[1].dtype == np.float16
        assert el.to_numpy().dtype == np.float16
        np.testing.assert_array_equal(el.to_numpy(), [[1.5, -2.5], [3.5, -4.5]])

    def test_add_chaining(self):
        el = EmbeddingList()
        result = el.add(np.array([1.0, 2.0])).add(np.array([3.0, 4.0]))
        assert result is el
        assert len(el) == 2


class TestEmbeddingListAddBatch:
    """Tests for add_batch()."""

    @pytest.mark.parametrize(
        "embeddings,more_embeddings",
        [
            pytest.param(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0]], dtype=np.float32),
                id="2d-array",
            ),
            pytest.param(
                [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                [np.array([5.0, 6.0])],
                id="list-of-arrays",
            ),
        ],
    )
    def test_add_batch_appends_in_order(self, embeddings, more_embeddings):
        el = EmbeddingList()
        result = el.add_batch(embeddings)
        assert result is el
        assert len(el) == 2
        np.testing.assert_array_equal(el.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])

        result = el.add_batch(more_embeddings)
        assert result is el
        assert len(el) == 3
        np.testing.assert_array_equal(el.to_numpy(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_add_batch_non_2d_raises(self):
        el = EmbeddingList()
        with pytest.raises(ValueError, match="2D"):
            el.add_batch(np.array([1.0, 2.0]))


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
    assert len(el) == count
    assert el.dim == dim
    assert el.to_numpy().shape == (count, dim)
    assert el.to_numpy().dtype == expected_dtype


def test_from_random_unsupported_dtype_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        EmbeddingList._from_random_test(2, 4, dtype="complex64")


def test_from_random_seed_is_reproducible():
    first = EmbeddingList._from_random_test(3, 4, seed=0)
    second = EmbeddingList._from_random_test(3, 4, seed=0)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


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
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])

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
        assert el.is_empty
        assert el.nbytes == 0
        assert el.total_dim == 0

        el.add([3.0, 4.0])
        np.testing.assert_array_equal(el.to_numpy(), [[3.0, 4.0]])


class TestEmbeddingListProperties:
    """Tests for all properties and magic methods."""

    def test_iter(self):
        el = EmbeddingList()
        el.add(np.array([1.0, 2.0]))
        el.add(np.array([3.0, 4.0]))
        items = list(el)
        assert len(items) == 2
        np.testing.assert_array_equal(items[0], [1.0, 2.0])
        np.testing.assert_array_equal(items[1], [3.0, 4.0])

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
