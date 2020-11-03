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


class BaseError(Exception):

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


class CollectionExistException(BaseError):
    def __init__(self, code, message):
        super().__init__(code, message)


class CollectionNotExistException(BaseError):
    def __init__(self, code, message):
        super().__init__(code, message)


class InvalidDimensionException(BaseError):
    def __init__(self, code, message):
        super().__init__(code, message)


class InvalidMetricTypeException(BaseError):
    def __init__(self, code, message):
        super().__init__(code, message)


class IllegalCollectionNameException(BaseError):
    def __init__(self, code, message):
        super().__init__(code, message)
