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
