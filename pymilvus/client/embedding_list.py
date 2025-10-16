"""EmbeddingList: A container for multiple embeddings for array-of-vector searches."""

from typing import Any, List, Optional, Union

import numpy as np

from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError


class EmbeddingList:
    """
    A container for multiple embeddings that can be used directly in Milvus searches.
    Represents a single query containing multiple vectors for array-of-vector fields.

    This is particularly useful for searching struct array fields that contain vectors,
    enabling array-of-vector to array-of-vector searches.

    Examples:
        >>> # Create empty and add vectors
        >>> query1 = EmbeddingList()
        >>> query1.add(embedding1)
        >>> query1.add(embedding2)
        >>>
        >>> # Create from list of vectors
        >>> vectors = [vec1, vec2, vec3]
        >>> query2 = EmbeddingList(vectors)
        >>>
        >>> # Use in search
        >>> results = client.search(
        >>>     collection_name="my_collection",
        >>>     data=[query1, query2],  # List of EmbeddingList
        >>>     ...
        >>> )
    """

    def __init__(
        self,
        embeddings: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        dim: Optional[int] = None,
        dtype: Optional[Union[np.dtype, str, DataType]] = None,
    ):
        """
        Initialize an EmbeddingList.

        Args:
            embeddings: Initial embeddings. Can be:
                - None: Creates an empty list
                - np.ndarray with shape (n, dim): Batch of n embeddings
                - np.ndarray with shape (dim,): Single embedding
                - List[np.ndarray]: List of embedding arrays
            dim: Expected dimension for validation (optional).
                 If provided, all added embeddings will be validated against this dimension.
            dtype: Data type of the embeddings (optional). Can be:
                - numpy dtype (e.g., np.float32, np.float16, np.uint8)
                - string (e.g., 'float32', 'float16', 'uint8')
                - DataType enum (e.g., DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR)
                If not specified, will infer from the first embedding added.
        """
        self._embeddings: List[np.ndarray] = []
        self._dim = dim
        self._dtype = self._parse_dtype(dtype) if dtype is not None else None

        if embeddings is not None:
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    # Single vector
                    self.add(embeddings)
                elif embeddings.ndim == 2:
                    # Multiple vectors
                    for i in range(len(embeddings)):
                        self.add(embeddings[i])
                else:
                    msg = "Embeddings array must be 1D or 2D"
                    raise ValueError(msg)
            elif isinstance(embeddings, list):
                for emb in embeddings:
                    self.add(emb)
            else:
                msg = "Embeddings must be numpy array or list"
                raise TypeError(msg)

    def _parse_dtype(self, dtype: Union[np.dtype, str, DataType]) -> np.dtype:
        """Parse and validate data type."""
        if isinstance(dtype, np.dtype):
            return dtype
        if isinstance(dtype, str):
            return np.dtype(dtype)
        if isinstance(dtype, DataType):
            # Map DataType enum to numpy dtype
            dtype_map = {
                DataType.FLOAT_VECTOR: np.float32,
                DataType.FLOAT16_VECTOR: np.float16,
                DataType.BFLOAT16_VECTOR: np.float16,  # Use float16 as approximation
                DataType.BINARY_VECTOR: np.uint8,
                DataType.INT8_VECTOR: np.int8,
            }
            if dtype in dtype_map:
                return np.dtype(dtype_map[dtype])
            msg = f"Unsupported DataType: {dtype}"
            raise ParamError(message=msg)
        msg = f"dtype must be numpy dtype, string, or DataType, got {type(dtype)}"
        raise TypeError(msg)

    def _infer_dtype(self, array: np.ndarray) -> np.dtype:
        """Infer dtype from array, with smart defaults."""
        if array.dtype == np.float64:
            # Default double precision to single precision for efficiency
            return np.dtype(np.float32)
        return array.dtype

    def add(self, embedding: Union[np.ndarray, List[Any]]) -> "EmbeddingList":
        """
        Add a single embedding vector to the list.

        Args:
            embedding: A single embedding vector (1D array or list)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If embedding dimension doesn't match existing embeddings
        """
        embedding = np.asarray(embedding)

        if embedding.ndim != 1:
            msg = f"Embedding must be 1D, got shape {embedding.shape}"
            raise ValueError(msg)

        # Validate dimension
        if self._embeddings:
            if len(embedding) != self.dim:
                msg = f"Embedding dimension {len(embedding)} doesn't match existing {self.dim}"
                raise ValueError(msg)
        elif self._dim is not None and len(embedding) != self._dim:
            msg = f"Embedding dimension {len(embedding)} doesn't match expected {self._dim}"
            raise ValueError(msg)

        # Handle dtype
        if self._dtype is None:
            # Infer dtype from first embedding
            self._dtype = self._infer_dtype(embedding)

        # Convert to the established dtype
        if embedding.dtype != self._dtype:
            embedding = embedding.astype(self._dtype)

        self._embeddings.append(embedding)
        return self

    def add_batch(self, embeddings: Union[List[np.ndarray], np.ndarray]) -> "EmbeddingList":
        """
        Add multiple embeddings at once.

        Args:
            embeddings: Batch of embeddings (2D array or list of 1D arrays)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If embeddings have inconsistent dimensions
        """
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim != 2:
                msg = f"Batch embeddings must be 2D, got {embeddings.ndim}D"
                raise ValueError(msg)
            for i in range(len(embeddings)):
                self.add(embeddings[i])
        else:
            for emb in embeddings:
                self.add(emb)
        return self

    @classmethod
    def _from_random_test(
        cls,
        num_vectors: int,
        dim: int,
        dtype: Optional[Union[np.dtype, str, DataType]] = None,
        seed: Optional[int] = None,
    ) -> "EmbeddingList":
        """
        Create an EmbeddingList with random vectors for testing purposes.

        WARNING: This method is intended for testing and demonstration only.
        Do not use in production code.

        Args:
            num_vectors: Number of random vectors to generate
            dim: Dimension of each vector
            dtype: Data type of the vectors (default: np.float32)
            seed: Random seed for reproducibility

        Returns:
            New EmbeddingList with random test vectors
        """
        rng = np.random.default_rng(seed)

        # Default dtype to float32 if None
        if dtype is None:
            dtype = np.dtype(np.float32)

        # Parse dtype if needed
        if not isinstance(dtype, np.dtype):
            dtype = cls(None, dim=dim, dtype=dtype)._dtype

        # Generate random data based on dtype
        if dtype == np.uint8:
            # For binary vectors, generate random bits
            embeddings = rng.integers(0, 256, size=(num_vectors, dim), dtype=np.uint8)
        elif dtype == np.int8:
            # For int8 vectors
            embeddings = rng.integers(-128, 128, size=(num_vectors, dim), dtype=np.int8)
        elif dtype in [np.float16, np.float32, np.float64]:
            # For float vectors
            embeddings = rng.random((num_vectors, dim)).astype(dtype)
        else:
            msg = f"Unsupported dtype for random generation: {dtype}"
            raise ValueError(msg)

        return cls(embeddings, dim=dim, dtype=dtype)

    def to_flat_array(self) -> np.ndarray:
        """
        Convert to flat array format required by Milvus for array-of-vector fields.

        Returns:
            Flattened numpy array containing all embeddings concatenated

        Raises:
            ValueError: If the list is empty
        """
        if not self._embeddings:
            msg = "EmbeddingList is empty"
            raise ValueError(msg)
        # Preserve the dtype of the embeddings
        return np.concatenate(self._embeddings)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to 2D numpy array.

        Returns:
            2D numpy array with shape (num_embeddings, dim)

        Raises:
            ValueError: If the list is empty
        """
        if not self._embeddings:
            msg = "EmbeddingList is empty"
            raise ValueError(msg)
        return np.stack(self._embeddings)

    def clear(self) -> "EmbeddingList":
        """Clear all embeddings from the list."""
        self._embeddings.clear()
        return self

    def __len__(self) -> int:
        """Return the number of embeddings in the list."""
        return len(self._embeddings)

    def __getitem__(self, index: int) -> np.ndarray:
        """Get embedding at specific index."""
        return self._embeddings[index]

    def __iter__(self):
        """Iterate over embeddings."""
        return iter(self._embeddings)

    @property
    def dim(self) -> int:
        """Dimension of each embedding."""
        if self._embeddings:
            return len(self._embeddings[0])
        return self._dim or 0

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Data type of the embeddings."""
        return self._dtype

    @property
    def shape(self) -> tuple:
        """Shape as (num_embeddings, dim)."""
        return (len(self), self.dim)

    @property
    def total_dim(self) -> int:
        """Total dimension of all embeddings combined."""
        return len(self) * self.dim

    @property
    def is_empty(self) -> bool:
        """Check if the list is empty."""
        return len(self._embeddings) == 0

    @property
    def nbytes(self) -> int:
        """Total number of bytes used by all embeddings."""
        if self.is_empty:
            return 0
        return sum(emb.nbytes for emb in self._embeddings)

    def __repr__(self) -> str:
        dtype_str = f", dtype={self.dtype}" if self.dtype else ""
        return f"EmbeddingList(count={len(self)}, dim={self.dim}{dtype_str})"

    def __str__(self) -> str:
        if self.is_empty:
            return "EmbeddingList(empty)"
        dtype_str = f" ({self.dtype})" if self.dtype else ""
        return f"EmbeddingList with {len(self)} embeddings of dimension {self.dim}{dtype_str}"
