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


class DeprecatedError(AttributeError):
    """
    API is deprecated
    """


class FutureTimeoutError(TimeoutError):
    """
    Future timeout
    """