# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

from enum import IntEnum

from .grpc_gen import common_pb2


class ErrorCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    RATE_LIMIT = 8
    FORCE_DENY = 9
    COLLECTION_NOT_FOUND = 100
    INDEX_NOT_FOUND = 700


class MilvusException(Exception):
    def __init__(
        self,
        code: int = ErrorCode.UNEXPECTED_ERROR,
        message: str = "",
        compatible_code: int = common_pb2.UnexpectedError,
    ) -> None:
        super().__init__()
        self._code = code
        self._message = message
        # for compatibility, remove it after 2.4.0
        self._compatible_code = compatible_code

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    @property
    def compatible_code(self):
        return self._compatible_code

    def __str__(self) -> str:
        return f"<{type(self).__name__}: (code={self.code}, message={self.message})>"


class ParamError(MilvusException):
    """Raise when params are incorrect"""


class ConnectError(MilvusException):
    """Connect server fail"""


class MilvusUnavailableException(MilvusException):
    """Raise when server's Unavaliable"""


class CollectionNotExistException(MilvusException):
    """Raise when collections doesn't exist"""


class DescribeCollectionException(MilvusException):
    """Raise when fail to describe collection"""


class PartitionAlreadyExistException(MilvusException):
    """Raise when create an exsiting partition"""


class IndexNotExistException(MilvusException):
    """Raise when index doesn't exist"""


class AmbiguousIndexName(MilvusException):
    """Raise multiple index exist, need specify index_name"""


class CannotInferSchemaException(MilvusException):
    """Raise when cannot trasfer dataframe to schema"""


class SchemaNotReadyException(MilvusException):
    """Raise when schema is wrong"""


class DataTypeNotMatchException(MilvusException):
    """Raise when datatype dosen't match"""


class DataTypeNotSupportException(MilvusException):
    """Raise when datatype isn't supported"""


class DataNotMatchException(MilvusException):
    """Raise when insert data isn't match with schema"""


class ConnectionNotExistException(MilvusException):
    """Raise when connections doesn't exist"""


class ConnectionConfigException(MilvusException):
    """Raise when configs of connection are invalid"""


class PrimaryKeyException(MilvusException):
    """Raise when primarykey are invalid"""


class PartitionKeyException(MilvusException):
    """Raise when partitionkey are invalid"""


class ClusteringKeyException(MilvusException):
    """Raise when clusteringkey are invalid"""


class FieldsTypeException(MilvusException):
    """Raise when fields is invalid"""


class FieldTypeException(MilvusException):
    """Raise when one field is invalid"""


class AutoIDException(MilvusException):
    """Raise when autoID is invalid"""


class InvalidConsistencyLevel(MilvusException):
    """Raise when consistency level is invalid"""


class UpsertAutoIDTrueException(MilvusException):
    """Raise when upsert autoID is true"""


class ExceptionsMessage:
    NoHostPort = "connection configuration must contain 'host' and 'port'."
    HostType = "Type of 'host' must be str."
    PortType = "Type of 'port' must be str or int."
    ConnDiffConf = (
        "Alias of %r already creating connections, "
        "but the configure is not the same as passed in."
    )
    AliasType = "Alias should be string, but %r is given."
    ConnLackConf = "You need to pass in the configuration of the connection named %r ."
    ConnectFirst = "should create connection first."
    CollectionNotExistNoSchema = "Collection %r not exist, or you can pass in schema to create one."
    NoSchema = "Should be passed into the schema."
    EmptySchema = "The field of the schema cannot be empty."
    SchemaType = "Schema type must be schema.CollectionSchema."
    SchemaInconsistent = (
        "The collection already exist, but the schema is not the same as the schema passed in."
    )
    AutoIDWithData = "Auto_id is True, primary field should not have data."
    AutoIDType = "Param auto_id must be bool type."
    NumPartitionsType = "Param num_partitions must be int type."
    AutoIDInconsistent = (
        "The auto_id of the collection is inconsistent "
        "with the auto_id of the primary key field."
    )
    AutoIDIllegalRanges = "The auto-generated id ranges should be pairs."
    ConsistencyLevelInconsistent = (
        "The parameter consistency_level is inconsistent with that of existed collection."
    )
    AutoIDOnlyOnPK = "The auto_id can only be specified on the primary key field"
    AutoIDFieldType = (
        "The auto_id can only be specified on field with DataType.INT64 and DataType.VARCHAR."
    )
    NumberRowsInvalid = "Must pass in at least one column"
    FieldsNumInconsistent = "The data fields number is not match with schema."
    NoVector = "No vector field is found."
    NoneDataFrame = "Dataframe can not be None."
    DataFrameType = "Data type must be pandas.DataFrame."
    NoPrimaryKey = "Schema must have a primary key field."
    PrimaryKeyNotExist = "Primary field must in dataframe."
    PrimaryKeyOnlyOne = "Expected only one primary key field, got [%s, %s, ...]."
    PartitionKeyOnlyOne = "Expected only one partition key field, got [%s, %s, ...]."
    PrimaryKeyType = "Primary key type must be DataType.INT64 or DataType.VARCHAR."
    PartitionKeyType = "Partition key field type must be DataType.INT64 or DataType.VARCHAR."
    PartitionKeyNotPrimary = "Partition key field should not be primary field"
    IsPrimaryType = "Param is_primary must be bool type."
    PrimaryFieldType = "Param primary_field must be int or str type."
    PartitionKeyFieldType = "Param partition_key_field must be str type."
    PartitionKeyFieldNotExist = "the specified partition key field {%s} not exist"
    IsPartitionKeyType = "Param is_partition_key must be bool type."
    DataTypeInconsistent = (
        "The Input data type is inconsistent with defined schema, please check it."
    )
    FieldDataInconsistent = "The Input data type is inconsistent with defined schema, {%s} field should be a %s, but got a {%s} instead."
    DataTypeNotSupport = "Data type is not support."
    DataLengthsInconsistent = "Arrays must all be same length."
    DataFrameInvalid = "Cannot infer schema from empty dataframe."
    NdArrayNotSupport = "Data type not support numpy.ndarray."
    TypeOfDataAndSchemaInconsistent = "The types of schema and data do not match."
    PartitionAlreadyExist = "Partition already exist."
    IndexNotExist = "Index doesn't exist."
    CollectionType = "The type of collection must be pymilvus.Collection."
    FieldsType = "The fields of schema must be type list."
    FieldType = "The field of schema type must be FieldSchema."
    FieldDtype = "Field dtype must be of DataType"
    ExprType = "The type of expr must be string ,but %r is given."
    EnvConfigErr = "Environment variable %s has a wrong format, please check it: %s"
    AmbiguousIndexName = "There are multiple indexes, please specify the index_name."
    InsertUnexpectedField = (
        "Attempt to insert an unexpected field `%s` to collection without enabling dynamic field"
    )
    UpsertAutoIDTrue = "Upsert don't support autoid == true"
    AmbiguousDeleteFilterParam = (
        "Ambiguous filter parameter, only one deletion condition can be specified."
    )
    AmbiguousQueryFilterParam = (
        "Ambiguous parameter, either ids or filter should be specified, cannot support both."
    )
    JSONKeyMustBeStr = "JSON key must be str."
    ClusteringKeyType = (
        "Clustering key field type must be DataType.INT8, DataType.INT16, "
        "DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE, "
        "DataType.VARCHAR, DataType.FLOAT_VECTOR."
    )
    ClusteringKeyFieldNotExist = "the specified clustering key field {%s} not exist"
    ClusteringKeyOnlyOne = "Expected only one clustering key field, got [%s, %s, ...]."
    IsClusteringKeyType = "Param is_clustering_key must be bool type."
    ClusteringKeyFieldType = "Param clustering_key_field must be str type."
