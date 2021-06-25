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


class ParamError(ValueError):
    """
    Param of interface is illegal
    """


class ConnectError(ValueError):
    """
    Connect server failed
    """


class NotConnectError(ConnectError):
    """
    Disconnect error
    """


class RepeatingConnectError(ConnectError):
    """
    Try to connect repeatedly
    """


class ConnectionPoolError(ConnectError):
    """
    Waiting timeout error
    """


class FutureTimeoutError(TimeoutError):
    """
    Future timeout
    """


class DeprecatedError(AttributeError):
    """
    Deprecated
    """


class VersionError(AttributeError):
    """
    Version not match
    """


class MilvusException(Exception):

    def __init__(self, code, message):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    def __str__(self):
        return f"<{type(self).__name__}: (code={self._code}, message={self._message})>"


class CollectionExistException(MilvusException):
    pass


class CollectionNotExistException(MilvusException):
    pass


class InvalidDimensionException(MilvusException):
    pass


class InvalidMetricTypeException(MilvusException):
    pass


class IllegalCollectionNameException(MilvusException):
    pass


class DescribeCollectionException(MilvusException):
    pass


class PartitionNotExistException(MilvusException):
    pass


class PartitionAlreadyExistException(MilvusException):
    pass


class InvalidArgumentException(MilvusException):
    pass


class IndexConflictException(MilvusException):
    pass


class IndexNotExistException(MilvusException):
    pass


class CannotInferSchemaException(MilvusException):
    pass


class SchemaNotReadyException(MilvusException):
    pass


class DataTypeNotMatchException(MilvusException):
    pass


class DataTypeNotSupportException(MilvusException):
    pass


class DataNotMatchException(MilvusException):
    pass


class ConnectionNotExistException(MilvusException):
    pass


class ConnectionConfigException(MilvusException):
    pass


class PrimaryKeyException(MilvusException):
    pass


class FieldsTypeException(MilvusException):
    pass


class FieldTypeException(MilvusException):
    pass


class AutoIDException(MilvusException):
    pass


class ExceptionsMessage:
    NoHostPort = "connection configuration must contain 'host' and 'port'."
    HostType = "Type of 'host' must be str."
    PortType = "Type of 'port' must be str or int."
    ConnDiffConf = "Alias of %r already creating connections, but the configure is not the same as passed in."
    AliasType = "Alias should be string, but %r is given."
    ConnLackConf = "You need to pass in the configuration of the connection named %r ."
    ConnectFirst = "should create connect first."
    NoSchema = "Should be passed into the schema."
    EmptySchema = "The field of the schema cannot be empty."
    SchemaType = "Schema type must be schema.CollectionSchema."
    SchemaInconsistent = "The collection already exist, but the schema is not the same as the schema passed in."
    AutoIDWithData = "Auto_id is True, primary field should not have data."
    AutoIDType = "Param auto_id must be bool type."
    AutoIDInconsistent = "The auto_id of the collection is inconsistent with the auto_id of the primary key field."
    AutoIDOnlyOnPK = "The auto_id can only be specified on the primary key field"
    FieldsNumInconsistent = "The data fields number is not match with schema."
    NoVector = "No vector field is found."
    NoneDataFrame = "Dataframe can not be None."
    DataFrameType = "Data type must be pandas.DataFrame."
    NoPrimaryKey = "Schema must have a primary key field."
    PrimaryKeyNotExist = "Primary field must in dataframe."
    PrimaryKeyOnlyOne = "Primary key field can only be one."
    PrimaryKeyType = "Primary key type must be DataType.INT64."
    IsPrimaryType = "Param is_primary must be bool type."
    DataTypeInconsistent = "The data in the same column must be of the same type."
    DataTypeNotSupport = "Data type is not support."
    DataLengthsInconsistent = "Arrays must all be same length."
    DataFrameInvalid = "Cannot infer schema from empty dataframe."
    NdArrayNotSupport = "Data type not support numpy.ndarray."
    TypeOfDataAndSchemaInconsistent = "The types of schema and data do not match."
    PartitionAlreadyExist = "Partition already exist."
    PartitionNotExist = "Partition not exist."
    IndexNotExist = "Index doesn't exist."
    CollectionType = "The type of collection must be pymilvus_orm.Collection."
    FieldsType = "The fields of schema must be type list."
    FieldType = "The field of schema type must be FieldSchema."
    FieldDtype = "Field dtype must be of DataType"
    ExprType = "The type of expr must be string ,but %r is given."
