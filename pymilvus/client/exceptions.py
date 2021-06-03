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


class BaseException(Exception):

    def __init__(self, code, message):
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
    def __init__(self, code, message):
        super().__init__(code, message)


class CollectionNotExistException(BaseException):
    def __init__(self, code, message):
        super().__init__(code, message)


class InvalidDimensionException(BaseException):
    def __init__(self, code, message):
        super().__init__(code, message)


class InvalidMetricTypeException(BaseException):
    def __init__(self, code, message):
        super().__init__(code, message)


class IllegalCollectionNameException(BaseException):
    def __init__(self, code, message):
        super().__init__(code, message)


class DescribeCollectionException(BaseException):
    def __init__(self, code, message):
        super().__init__(code, message)

