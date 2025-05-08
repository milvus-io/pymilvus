from typing import Any, Iterable, Optional, Tuple, TypeVar, Union, get_args, get_origin

from pymilvus.exceptions import ParamError

T = TypeVar("T")


def validate_params(**params: Tuple[Any, Any, Optional[bool]]) -> None:
    """
    Batch-validate multiple parameters via keyword arguments.

    Each keyword name is the parameter name, and its value is a tuple:
        (value, expected_type[, noneable])

    Args:
        **params: Keyword specs for parameters to validate.

    Raises:
        ParamError: If any spec tuple is invalid or any parameter fails validation.

    Examples:
        validate_params(
            ids=([1, 2, 3], List[int]),
            score=(score_value, float, True),
        )
    """
    for name, spec in params.items():
        if not (2 <= len(spec) <= 3):
            msg = f"Validation spec for '{name}' must be (value, expected_type[, noneable])."
            raise ParamError(message=msg)

        value, expected_type = spec[0], spec[1]
        noneable = spec[2] if len(spec) == 3 else False
        validate_param(name, value, expected_type, noneable=noneable)


def validate_param(param_name: str, param: Any, expected_type: Any, noneable: bool = False) -> None:
    """
    Validate a single parameter's type, including generic iterable subtypes.

    Args:
        param_name (str): The name of the parameter (for error messages).
        param (Any): The value to validate.
        expected_type (Any): Expected type or typing.Generic (e.g., List[int]).
        noneable (bool, optional): If True, allows `param` to be None without error.
            Defaults to False.

    Raises:
        ParamError: If `param` is None (when noneable is False), not of `expected_type`,
            or contains elements not matching generic subtypes.
    """
    # Handle None
    if param is None:
        if noneable:
            return
        raise ParamError(message=f"missing required argument: {param_name}")

    # Capture generic subtype args
    origin = get_origin(expected_type)
    subtype_args = get_args(expected_type) if origin else None
    base_type = origin or expected_type

    # Check main type
    if not isinstance(param, base_type):
        _raise_param_error(param_name, expected_type, param)

    # Check element types if iterable generic with one subtype, ignore str and bytes
    if subtype_args and isinstance(param, Iterable) and not isinstance(param, (str, bytes)):
        # Only support single subtype, e.g., List[int]
        if len(subtype_args) != 1:
            raise ParamError(message=f"Unsupported generic iterable type: {expected_type}.")

        subtype = subtype_args[0]
        for idx, item in enumerate(param):
            if not isinstance(item, subtype):
                msg = (
                    f"wrong type of element {idx} of [{param_name}], "
                    f"expected type [{subtype.__name__}], got [{type(item).__name__}]."
                )
                raise ParamError(message=msg)


def _type_name(tp: Union[type, Tuple[type, ...]]) -> str:
    if isinstance(tp, tuple):
        return ", ".join(t.__name__ for t in tp)
    return tp.__name__


def _raise_param_error(param_name: str, expected: Union[type, Tuple[type, ...]], actual: Any):
    expected_name = _type_name(expected)
    actual_name = type(actual).__name__

    msg = (
        f"wrong type of argument [{param_name}], "
        f"expected type [{expected_name}], got [{actual_name}]."
    )
    raise ParamError(message=msg)
