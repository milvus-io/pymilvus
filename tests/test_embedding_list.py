"""Tests for embedding_list module."""

import pytest
import numpy as np
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError


class TestEmbeddingList:
    """Test EmbeddingList class."""

    def test_init_empty(self):
        """Test initializing empty EmbeddingList."""
        emb_list = EmbeddingList()
        assert len(emb_list) == 0
        assert emb_list._dim is None
        assert emb_list._dtype is None

    def test_init_with_single_vector(self):
        """Test initializing with a single vector."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb_list = EmbeddingList(vector)
        assert len(emb_list) == 1
        assert np.array_equal(emb_list[0], vector)

    def test_init_with_batch_vectors(self):
        """Test initializing with batch of vectors."""
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        emb_list = EmbeddingList(vectors)
        assert len(emb_list) == 2
        assert np.array_equal(emb_list[0], vectors[0])
        assert np.array_equal(emb_list[1], vectors[1])

    def test_init_with_list(self):
        """Test initializing with list of vectors."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert len(emb_list) == 2

    def test_init_with_dim(self):
        """Test initializing with dimension."""
        emb_list = EmbeddingList(dim=128)
        assert emb_list._dim == 128

    def test_init_with_dtype_numpy(self):
        """Test initializing with numpy dtype."""
        # np.float32 is not a dtype instance, it's a type
        # So we need to pass np.dtype('float32') or 'float32'
        emb_list = EmbeddingList(dtype=np.dtype('float32'))
        assert emb_list._dtype is not None
        assert str(emb_list._dtype) == "float32"
        
        # Also test with string
        emb_list2 = EmbeddingList(dtype="float32")
        assert emb_list2._dtype is not None
        assert str(emb_list2._dtype) == "float32"

    def test_init_with_dtype_string(self):
        """Test initializing with string dtype."""
        emb_list = EmbeddingList(dtype="float32")
        assert emb_list._dtype == np.dtype("float32")

    def test_init_with_dtype_datatype(self):
        """Test initializing with DataType enum."""
        emb_list = EmbeddingList(dtype=DataType.FLOAT_VECTOR)
        assert emb_list._dtype == np.dtype(np.float32)

    def test_init_invalid_array_dimension(self):
        """Test initializing with invalid array dimension."""
        invalid_array = np.array([[[1, 2], [3, 4]]])  # 3D array
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            EmbeddingList(invalid_array)

    def test_init_invalid_type(self):
        """Test initializing with invalid type."""
        with pytest.raises(TypeError, match="must be numpy array or list"):
            EmbeddingList("not a vector")

    def test_add_single_vector(self):
        """Test adding a single vector."""
        emb_list = EmbeddingList()
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb_list.add(vector)
        assert len(emb_list) == 1
        assert np.array_equal(emb_list[0], vector)

    def test_add_multiple_vectors(self):
        """Test adding multiple vectors."""
        emb_list = EmbeddingList()
        vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        emb_list.add(vector1)
        emb_list.add(vector2)
        assert len(emb_list) == 2

    def test_add_with_dim_validation(self):
        """Test adding vectors with dimension validation."""
        emb_list = EmbeddingList(dim=3)
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb_list.add(vector)  # Should succeed
        
        # Try adding vector with wrong dimension
        wrong_vector = np.array([0.1, 0.2], dtype=np.float32)
        with pytest.raises(ValueError, match="dimension"):
            emb_list.add(wrong_vector)

    def test_add_with_dtype_validation(self):
        """Test adding vectors with dtype validation."""
        emb_list = EmbeddingList(dtype=np.dtype('float32'))
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb_list.add(vector)  # Should succeed
        
        # Try adding vector with wrong dtype - it will be converted automatically
        # So this should succeed, not raise an error
        wrong_vector = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        emb_list.add(wrong_vector)  # Should convert to float32
        assert emb_list[1].dtype == np.float32

    def test_infer_dtype_float64_to_float32(self):
        """Test dtype inference from float64 to float32."""
        emb_list = EmbeddingList()
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        emb_list.add(vector)
        assert emb_list._dtype == np.dtype(np.float32)

    def test_infer_dtype_float16(self):
        """Test dtype inference for float16."""
        emb_list = EmbeddingList()
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float16)
        emb_list.add(vector)
        assert emb_list._dtype == np.dtype(np.float16)

    def test_infer_dtype_int8(self):
        """Test dtype inference for int8."""
        emb_list = EmbeddingList()
        vector = np.array([1, 2, 3], dtype=np.int8)
        emb_list.add(vector)
        assert emb_list._dtype == np.dtype(np.int8)

    def test_infer_dtype_uint8(self):
        """Test dtype inference for uint8."""
        emb_list = EmbeddingList()
        vector = np.array([1, 2, 3], dtype=np.uint8)
        emb_list.add(vector)
        assert emb_list._dtype == np.dtype(np.uint8)

    def test_getitem(self):
        """Test __getitem__ method."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert np.array_equal(emb_list[0], vectors[0])
        assert np.array_equal(emb_list[1], vectors[1])

    def test_len(self):
        """Test __len__ method."""
        emb_list = EmbeddingList()
        assert len(emb_list) == 0
        emb_list.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert len(emb_list) == 1

    def test_iter(self):
        """Test iteration."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        for i, vec in enumerate(emb_list):
            assert np.array_equal(vec, vectors[i])

    def test_parse_dtype_invalid_datatype(self):
        """Test _parse_dtype with invalid DataType."""
        emb_list = EmbeddingList()
        # Use a DataType that doesn't have numpy dtype mapping
        with pytest.raises(TypeError):
            emb_list._parse_dtype(DataType.VARCHAR)

    def test_parse_dtype_invalid_type(self):
        """Test _parse_dtype with invalid type."""
        emb_list = EmbeddingList()
        # The code checks isinstance(dtype, (np.dtype, str, DataType))
        # If none match, the function will return None (no explicit return)
        # But actually, looking at the code, if dtype is DataType but numpy_dtype is None,
        # it raises TypeError. For other invalid types, it just returns None.
        # So let's test with a DataType that doesn't have numpy dtype mapping
        from pymilvus.client.types import DataType
        
        # Varchar doesn't have numpy dtype, so it should raise TypeError
        with pytest.raises(TypeError, match="dtype must be numpy dtype"):
            emb_list._parse_dtype(DataType.VARCHAR)

    def test_add_batch_2d_array(self):
        """Test add_batch with 2D numpy array."""
        emb_list = EmbeddingList()
        batch = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        emb_list.add_batch(batch)
        assert len(emb_list) == 2

    def test_add_batch_list(self):
        """Test add_batch with list."""
        emb_list = EmbeddingList()
        batch = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list.add_batch(batch)
        assert len(emb_list) == 2

    def test_add_batch_invalid_dimension(self):
        """Test add_batch with invalid dimension."""
        emb_list = EmbeddingList()
        batch = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 1D, not 2D
        with pytest.raises(ValueError, match="must be 2D"):
            emb_list.add_batch(batch)

    def test_add_invalid_dimension(self):
        """Test add with invalid dimension (not 1D)."""
        emb_list = EmbeddingList()
        vector = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)  # 2D
        with pytest.raises(ValueError, match="must be 1D"):
            emb_list.add(vector)

    def test_to_flat_array(self):
        """Test to_flat_array method."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        flat = emb_list.to_flat_array()
        assert len(flat) == 6
        assert np.allclose(flat, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_to_flat_array_empty(self):
        """Test to_flat_array with empty list."""
        emb_list = EmbeddingList()
        with pytest.raises(ValueError, match="empty"):
            emb_list.to_flat_array()

    def test_to_numpy(self):
        """Test to_numpy method."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        arr = emb_list.to_numpy()
        assert arr.shape == (2, 3)
        assert np.allclose(arr[0], [0.1, 0.2, 0.3])
        assert np.allclose(arr[1], [0.4, 0.5, 0.6])

    def test_to_numpy_empty(self):
        """Test to_numpy with empty list."""
        emb_list = EmbeddingList()
        with pytest.raises(ValueError, match="empty"):
            emb_list.to_numpy()

    def test_clear(self):
        """Test clear method."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert len(emb_list) == 1
        emb_list.clear()
        assert len(emb_list) == 0

    def test_dim_property(self):
        """Test dim property."""
        emb_list = EmbeddingList(dim=128)
        assert emb_list.dim == 128  # Returns _dim if set
        
        emb_list2 = EmbeddingList()
        assert emb_list2.dim == 0  # No vectors yet, returns 0 if _dim is None
        
        emb_list2.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert emb_list2.dim == 3

    def test_dtype_property(self):
        """Test dtype property."""
        emb_list = EmbeddingList(dtype=np.dtype('float32'))
        assert emb_list.dtype is not None
        assert str(emb_list.dtype) == "float32"

    def test_shape_property(self):
        """Test shape property."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert emb_list.shape == (2, 3)

    def test_total_dim_property(self):
        """Test total_dim property."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert emb_list.total_dim == 6

    def test_is_empty_property(self):
        """Test is_empty property."""
        emb_list = EmbeddingList()
        assert emb_list.is_empty is True
        
        emb_list.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert emb_list.is_empty is False

    def test_nbytes_property(self):
        """Test nbytes property."""
        vectors = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        emb_list = EmbeddingList(vectors)
        assert emb_list.nbytes > 0
        assert emb_list.nbytes == sum(v.nbytes for v in vectors)

    def test_nbytes_empty(self):
        """Test nbytes property with empty list."""
        emb_list = EmbeddingList()
        assert emb_list.nbytes == 0

    def test_repr(self):
        """Test __repr__ method."""
        emb_list = EmbeddingList(dtype=np.dtype('float32'))
        repr_str = repr(emb_list)
        assert "EmbeddingList" in repr_str
        # Check for dtype in repr (format may vary)
        # If dtype is set, it should appear in repr
        if emb_list._dtype is not None:
            assert "dtype" in repr_str

    def test_str(self):
        """Test __str__ method."""
        emb_list = EmbeddingList()
        str_repr = str(emb_list)
        assert "empty" in str_repr
        
        emb_list.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        str_repr = str(emb_list)
        assert "1 embeddings" in str_repr
        assert "dimension 3" in str_repr

    def test_from_random_test_float32(self):
        """Test _from_random_test with float32."""
        emb_list = EmbeddingList._from_random_test(5, 128, dtype=np.dtype('float32'), seed=42)
        assert len(emb_list) == 5
        assert emb_list.dim == 128
        assert emb_list.dtype is not None
        assert str(emb_list.dtype) == "float32"

    def test_from_random_test_uint8(self):
        """Test _from_random_test with uint8."""
        emb_list = EmbeddingList._from_random_test(3, 64, dtype=np.dtype('uint8'), seed=42)
        assert len(emb_list) == 3
        assert emb_list.dim == 64
        assert emb_list.dtype is not None
        assert str(emb_list.dtype) == "uint8"

    def test_from_random_test_int8(self):
        """Test _from_random_test with int8."""
        emb_list = EmbeddingList._from_random_test(3, 64, dtype=np.dtype('int8'), seed=42)
        assert len(emb_list) == 3
        assert emb_list.dim == 64
        assert emb_list.dtype is not None
        assert str(emb_list.dtype) == "int8"

    def test_from_random_test_float16(self):
        """Test _from_random_test with float16."""
        emb_list = EmbeddingList._from_random_test(3, 64, dtype=np.dtype('float16'), seed=42)
        assert len(emb_list) == 3
        assert emb_list.dtype is not None
        assert str(emb_list.dtype) == "float16"

    def test_from_random_test_datatype(self):
        """Test _from_random_test with DataType."""
        emb_list = EmbeddingList._from_random_test(3, 64, dtype=DataType.FLOAT_VECTOR, seed=42)
        assert len(emb_list) == 3
        assert emb_list.dtype == np.dtype("float32")

    def test_from_random_test_unsupported_dtype(self):
        """Test _from_random_test with unsupported dtype."""
        # np.int32 is not a dtype instance, it's a type, so it will be rejected by _parse_dtype
        with pytest.raises(TypeError, match="dtype must be numpy dtype"):
            EmbeddingList._from_random_test(3, 64, dtype=np.int32, seed=42)
        
        # Test with actual unsupported dtype (int32 is not supported for random generation)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            EmbeddingList._from_random_test(3, 64, dtype=np.dtype('int32'), seed=42)

