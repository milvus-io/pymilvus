from enum import IntEnum


class Status:
    """
    :attribute code: int (optional) default as ok

    :attribute message: str (optional) current status message
    """

    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    CONNECT_FAILED = 2
    PERMISSION_DENIED = 3
    COLLECTION_NOT_EXISTS = 4
    ILLEGAL_ARGUMENT = 5
    ILLEGAL_RANGE = 6
    ILLEGAL_DIMENSION = 7
    ILLEGAL_INDEX_TYPE = 8
    ILLEGAL_COLLECTION_NAME = 9
    ILLEGAL_TOPK = 10
    ILLEGAL_ROWRECORD = 11
    ILLEGAL_VECTOR_ID = 12
    ILLEGAL_SEARCH_RESULT = 13
    FILE_NOT_FOUND = 14
    META_FAILED = 15
    CACHE_FAILED = 16
    CANNOT_CREATE_FOLDER = 17
    CANNOT_CREATE_FILE = 18
    CANNOT_DELETE_FOLDER = 19
    CANNOT_DELETE_FILE = 20
    BUILD_INDEX_ERROR = 21
    ILLEGAL_NLIST = 22
    ILLEGAL_METRIC_TYPE = 23
    OUT_OF_MEMORY = 24

    def __init__(self, code=SUCCESS, message="Success"):
        self.code = code
        self.message = message

    def __repr__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __eq__(self, other):
        """
        Make Status comparable with self by code
        """
        if isinstance(other, int):
            return self.code == other

        return isinstance(other, self.__class__) and self.code == other.code

    def __ne__(self, other):
        return self != other

    def OK(self):
        return self.code == Status.SUCCESS


class IndexType(IntEnum):
    """Index type enum.
    """

    #: Invalid index type.
    INVALID = 0

    #: FLAT index. See `FLAT <https://milvus.io/docs/v1.0.0/index.md#FLAT>`_.
    FLAT = 1

    #: IVF(Inverted File) FLAT index.
    #: See `IVF_FLAT <https://milvus.io/docs/v1.0.0/index.md#IVF_FLAT>`_.
    IVF_FLAT = 2

    #: IVF SQ8 index. See `IVF_SQ8 <https://milvus.io/docs/v1.0.0/index.md#IVF_SQ8>`_.
    IVF_SQ8 = 3

    #: RNSG(Refined NSG) index. See `RNSG <https://milvus.io/docs/v1.0.0/index.md#RNSG>`_.
    RNSG = 4

    #: IVF SQ8 Hybrid index. See `IVF_SQ8H <https://milvus.io/docs/v1.0.0/index.md#IVF_SQ8H>`_.
    IVF_SQ8H = 5

    #: IVF PQ index. See `IVF_PQ <https://milvus.io/docs/v1.0.0/index.md#IVF_PQ>`_.
    IVF_PQ = 6

    #: HNSW index. See `HNSW <https://milvus.io/docs/v1.0.0/index.md#HNSW>`_.
    HNSW = 11

    #: ANNOY index. See `ANNOY <https://milvus.io/docs/v1.0.0/index.md#ANNOY>`_.
    ANNOY = 12

    #: Alternative name for `IVF_FLAT`. Reserved for compatibility.
    IVFLAT = IVF_FLAT
    #: Alternative name for `IVF_SQ8H`. Reserved for compatibility.
    IVF_SQ8_H = IVF_SQ8H

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_


class MetricType(IntEnum):
    """Metric type enum.
    """

    #: Invalid metric type.
    INVALID = 0

    #: Euclidean distance. A metric for float vectors.
    #: See `Euclidean distance <https://milvus.io/docs/v1.0.0/metric.md#Euclidean-distance-L2>`_.
    L2 = 1

    #: Inner product. A metric for float vectors.
    #: See `Inner Product <https://milvus.io/docs/v1.0.0/metric.md#Inner-product-IP>`_.
    IP = 2

    #: Hamming distance. A metric for binary vectors.
    #: See `Hamming distance <https://milvus.io/docs/v1.0.0/metric.md#Hamming-distance>`_.
    HAMMING = 3

    #: Jaccard distance. A metric for binary vectors.
    #: See `Jaccard distance <https://milvus.io/docs/v1.0.0/metric.md#Jaccard-distance>`_.
    JACCARD = 4

    #: Tanimoto distance. A metric for binary vectors.
    #: See `Tanimoto distance <https://milvus.io/docs/v1.0.0/metric.md#Tanimoto-distance>`_.
    TANIMOTO = 5

    #: Superstructure. A metric for binary vectors,
    #: only support :attr:`~milvus.IndexType.FLAT` index.
    #: See `Superstructure <https://milvus.io/docs/v1.0.0/metric.md#Superstructure>`_.
    SUBSTRUCTURE = 6

    #: Substructure. A metric for binary vectors, only support :attr:`~milvus.IndexType.FLAT` index.
    #: See `Substructure <https://milvus.io/docs/v1.0.0/metric.md#Substructure>`_.
    SUPERSTRUCTURE = 7

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_
