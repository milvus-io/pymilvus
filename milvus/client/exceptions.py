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


# TODO: rename this & change test cases use `BaseException`
# pylint: disable=redefined-builtin
class BaseException(Exception):

    def __init__(self, code, message):
        super(BaseException, self).__init__(code, message)
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


class CollectionExistException(BaseException):
    pass


class CollectionNotExistException(BaseException):
    pass


class InvalidDimensionException(BaseException):
    pass


class InvalidMetricTypeException(BaseException):
    pass


class IllegalCollectionNameException(BaseException):
    pass


class DescribeCollectionException(BaseException):
    pass
