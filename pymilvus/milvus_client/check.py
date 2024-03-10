from typing import Any


def check_param_type(param_name: str, param: Any, expected_type: Any, ignore_none: bool = True):
    if ignore_none and param is None:
        return
    if not isinstance(param, expected_type):
        msg = f"wrong type of arugment '{param_name}', "
        msg += f"expected '{expected_type.__name__}', "
        msg += f"got '{type(param).__name__}'"
        raise TypeError(msg)
