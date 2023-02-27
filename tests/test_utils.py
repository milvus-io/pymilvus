import pytest
from pymilvus.utils import TypeChecker
from pymilvus.exceptions import ParamError


class TestTypeChecker:
    @pytest.mark.parametrize("type_str", [
        "s",
        "_",
    ])
    def test_str_type(self, type_str):
        TypeChecker.check(
            collection_name=type_str,
            alias=type_str,
            name=type_str,
            partition_name=type_str,
            index_name=type_str,
            field_name=type_str,
            anns_field=type_str,
        )

    @pytest.mark.parametrize("not_type_str", [
        None,
        1,
        1.0,
        [1],
        {1: 2},
        {1, 2, 3},
        (1, 2, 3),
    ])
    def test_str_type_error(self, not_type_str):
        with pytest.raises(ParamError):
            TypeChecker.check(collection_name=not_type_str)

    @pytest.mark.parametrize("type_int", [
        1,
        0, 
        -1,
        1024,
        65536,
    ])
    def test_int_type(self, type_int):
        TypeChecker.check(
            round_decimal=type_int,
            num_replica=type_int,
            dim=type_int,
            limit=type_int,
            topk=type_int,
        )

    @pytest.mark.parametrize("not_type_int", [
        None,
        "abc",
        1.0,
        [1],
        {1: 2},
        {1, 2, 3},
        (1, 2, 3),
    ])
    def test_int_type_error(self, not_type_int):
        with pytest.raises(ParamError):
            TypeChecker.check(topk=not_type_int)

    @pytest.mark.parametrize("type_list_str", [
        ["a"],
        ["1", "2"]
    ])
    def test_list_str_type(self, type_list_str):
        TypeChecker.check(
            partition_names=type_list_str,
            output_fields=type_list_str,
        )

    @pytest.mark.parametrize("not_type_list_str", [
        None,
        "abc",
        1.0,
        {1: 2},
        {1, 2, 3},
        (1, 2, 3),
        [1],
        [],
        [1, "1"],
    ])
    def test_int_type_error(self, not_type_list_str):
        with pytest.raises(ParamError):
            TypeChecker.check(topk=not_type_list_str)
