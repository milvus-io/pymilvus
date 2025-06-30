import pytest
from typing import List, Optional

from pymilvus._utils.validator import validate_param, validate_params
from pymilvus.exceptions import ParamError


def test_validate_param_success_simple():
    # int passes
    validate_param("age", 30, int)
    # None when noneable=True passes
    validate_param("maybe", None, str, noneable=True)
    # Iterable subtype passes
    validate_param("ids", [1, 2, 3], List[int])


def test_validate_param_failure_none():
    with pytest.raises(ParamError) as exc:
        validate_param("name", None, str)
    assert "missing required argument:" in str(exc.value)


def test_validate_param_failure_type():
    with pytest.raises(ParamError):
        validate_param("flag", "yes", bool)


def test_validate_param_failure_subtype():
    with pytest.raises(ParamError) as exc:
        validate_param("chars", ["a", 2, "c"], List[str])
    assert "wrong type of element" in str(exc.value)


def test_validate_params_success():
    validate_params(
        age=(25, int),
        score=(None, float, True),
        tags=(["x", "y"], List[str])
    )


def test_validate_params_invalid_spec_length():
    with pytest.raises(ParamError):
        validate_params(x=(1, int, True, str))


def test_validate_params_invalid_container():
    with pytest.raises(TypeError):
        validate_params({})


def test_validate_params_failure_nested():
    with pytest.raises(ParamError) as exc:
        validate_params(nums=([1, "two"], List[int]))
    assert "expected type [int], got [str]" in str(exc.value)
