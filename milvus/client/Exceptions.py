class ParamError(ValueError):
    """
    Param of interface is illegal
    """
    pass


class ConnectError(ValueError):
    """
    Connect server failed
    """
    pass


class NotConnectError(ConnectError):
    """
    Disconnect error
    """
    pass


class RepeatingConnectError(ConnectError):
    pass


class DisconnectNotConnectedClientError(ConnectError):
    pass
