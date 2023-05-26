from pymilvus.exceptions import MilvusException


class CollectionDoesNotExist(MilvusException):
    """Collection doesnt exist"""


class CollectionAlreadyExists(MilvusException):
    """Collection already exists"""


class InvalidPartitionFieldFormat(MilvusException):
    """Partition fields invalid format"""


class InvalidDistanceMetric(MilvusException):
    """Invalid distnace metric supplied"""


class InvalidInsertBatchSize(MilvusException):
    """Invalid batch size for insert supplied"""


class InvalidPKFormat(MilvusException):
    """Invalid PK format"""
