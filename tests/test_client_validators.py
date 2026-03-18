import datetime

import numpy as np
import pytest
from pymilvus.client.check import (
    ParamChecker,
    check_id_and_data,
    check_pass_param,
    int_or_str,
    is_correct_date_str,
    is_legal_address,
    is_legal_anns_field,
    is_legal_cmd,
    is_legal_db_name,
    is_legal_dimension,
    is_legal_drop_ratio,
    is_legal_field_name,
    is_legal_guarantee_timestamp,
    is_legal_host,
    is_legal_ids,
    is_legal_index_name,
    is_legal_index_size,
    is_legal_limit,
    is_legal_nlist,
    is_legal_nprobe,
    is_legal_output_fields,
    is_legal_partition_name,
    is_legal_partition_name_array,
    is_legal_password,
    is_legal_port,
    is_legal_privilege,
    is_legal_privilege_group,
    is_legal_privileges,
    is_legal_replica_number,
    is_legal_role_name,
    is_legal_round_decimal,
    is_legal_table_name,
    is_legal_timeout,
    is_legal_topk,
    is_legal_user,
    parser_range_date,
    validate_nullable_strs,
    validate_strs,
)
from pymilvus.client.constants import BOUNDED_TS, EVENTUALLY_TS, GUARANTEE_TIMESTAMP, ITERATOR_FIELD
from pymilvus.client.ts_utils import construct_guarantee_ts
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.check import validate_noneable_param, validate_param, validate_params


class TestValidateStrs:
    def test_valid(self):
        validate_strs(name="hello")

    def test_empty_raises(self):
        with pytest.raises(ParamError):
            validate_strs(name="")

    def test_none_raises(self):
        with pytest.raises(ParamError):
            validate_strs(name=None)

    def test_int_raises(self):
        with pytest.raises(ParamError):
            validate_strs(name=123)


class TestValidateNullableStrs:
    def test_none_ok(self):
        validate_nullable_strs(name=None)

    def test_valid_str_ok(self):
        validate_nullable_strs(name="hello")

    def test_empty_str_raises(self):
        with pytest.raises(ParamError):
            validate_nullable_strs(name="")

    def test_int_raises(self):
        with pytest.raises(ParamError):
            validate_nullable_strs(name=42)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("localhost:19530", True),
        ("localhost", False),
        (1234, False),
        (":19530", False),
        ("localhost:abc", False),
    ],
)
def test_is_legal_address(value, expected):
    assert is_legal_address(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("localhost", True),
        ("", False),
        ("loc:al", False),
        (123, False),
    ],
)
def test_is_legal_host(value, expected):
    assert is_legal_host(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (19530, True),
        ("19530", True),
        ("abc", False),
        (3.14, False),
    ],
)
def test_is_legal_port(value, expected):
    assert is_legal_port(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (128, True),
        ("128", True),
        ("abc", False),
    ],
)
def test_is_legal_dimension(value, expected):
    assert is_legal_dimension(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ([1, 2, 3], True),
        (["a", "b"], True),
        ([], False),
        ([True], False),
        ([1, "a"], False),
        ([np.int64(1), np.int64(2)], True),
        ([2**64], False),
    ],
)
def test_is_legal_ids(value, expected):
    assert is_legal_ids(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        (10, True),
        (1.5, True),
        ("10", False),
    ],
)
def test_is_legal_timeout(value, expected):
    assert is_legal_timeout(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (10, True),
        (0, False),
        (-1, False),
        (1.5, False),
    ],
)
def test_is_legal_limit(value, expected):
    assert is_legal_limit(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, True),
        (6, True),
        (-1, True),
        (7, False),
        (-2, False),
    ],
)
def test_is_legal_round_decimal(value, expected):
    assert is_legal_round_decimal(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        (0, True),
        (100, True),
        (-1, False),
        ("100", False),
    ],
)
def test_is_legal_guarantee_timestamp(value, expected):
    assert is_legal_guarantee_timestamp(value) is expected


class TestIsLegalTableName:
    def test_valid(self):
        assert is_legal_table_name("col") is True

    def test_empty(self):
        assert not is_legal_table_name("")

    def test_none(self):
        assert not is_legal_table_name(None)


class TestIsLegalDbName:
    def test_empty_str_ok(self):
        assert is_legal_db_name("") is True

    def test_valid(self):
        assert is_legal_db_name("mydb") is True

    def test_none(self):
        assert is_legal_db_name(None) is False


class TestIsLegalOutputFields:
    def test_none(self):
        assert is_legal_output_fields(None) is True

    def test_valid_list(self):
        assert is_legal_output_fields(["a", "b"]) is True

    def test_not_list(self):
        assert is_legal_output_fields("a") is False

    def test_list_with_empty(self):
        assert is_legal_output_fields(["a", ""]) is False


class TestIsLegalPartitionName:
    def test_valid(self):
        assert is_legal_partition_name("part") is True

    def test_none(self):
        assert is_legal_partition_name(None) is False

    def test_empty(self):
        assert is_legal_partition_name("") is True


class TestIsLegalPartitionNameArray:
    def test_none(self):
        assert is_legal_partition_name_array(None) is True

    def test_valid(self):
        assert is_legal_partition_name_array(["p1", "p2"]) is True

    def test_not_list(self):
        assert is_legal_partition_name_array("p1") is False


class TestIsLegalDropRatio:
    def test_valid(self):
        assert is_legal_drop_ratio(0.5) is True

    def test_zero(self):
        assert is_legal_drop_ratio(0.0) is True

    def test_one(self):
        assert is_legal_drop_ratio(1.0) is False

    def test_negative(self):
        assert is_legal_drop_ratio(-0.1) is False

    def test_int(self):
        assert is_legal_drop_ratio(0) is False


class TestIsLegalPrivilege:
    def test_valid(self):
        assert is_legal_privilege("Insert") is True

    def test_empty(self):
        assert not is_legal_privilege("")

    def test_none(self):
        assert not is_legal_privilege(None)


class TestIsLegalPrivilegeGroup:
    def test_valid(self):
        assert is_legal_privilege_group("grp1") is True

    def test_empty(self):
        assert not is_legal_privilege_group("")


class TestIsLegalPrivileges:
    def test_valid(self):
        assert is_legal_privileges(["Insert", "Delete"]) is True

    def test_empty_list(self):
        assert not is_legal_privileges([])

    def test_contains_empty(self):
        assert not is_legal_privileges(["Insert", ""])

    def test_none(self):
        assert not is_legal_privileges(None)


class TestMiscValidators:
    def test_int_or_str_int(self):
        assert int_or_str(42) == "42"

    def test_int_or_str_str(self):
        assert int_or_str("hello") == "hello"

    def test_is_correct_date_str_always_false(self):
        assert is_correct_date_str("2024-01-01") is False

    def test_is_legal_cmd_valid(self):
        assert is_legal_cmd("status") is True

    def test_is_legal_cmd_empty(self):
        assert not is_legal_cmd("")

    def test_is_legal_user(self):
        assert is_legal_user("alice") is True
        assert is_legal_user(123) is False

    def test_is_legal_password(self):
        assert is_legal_password("secret") is True
        assert is_legal_password(None) is False

    def test_is_legal_role_name(self):
        assert is_legal_role_name("admin") is True
        assert not is_legal_role_name("")

    def test_is_legal_replica_number(self):
        assert is_legal_replica_number(1) is True
        assert is_legal_replica_number("1") is False

    def test_is_legal_index_size(self):
        assert is_legal_index_size(1024) is True
        assert is_legal_index_size("1024") is False

    def test_is_legal_topk(self):
        assert is_legal_topk(10) is True
        assert is_legal_topk(True) is False

    def test_is_legal_nlist(self):
        assert is_legal_nlist(128) is True
        assert is_legal_nlist(False) is False

    def test_is_legal_nprobe(self):
        assert is_legal_nprobe(10) is True
        assert is_legal_nprobe("10") is False

    def test_is_legal_index_name(self):
        assert is_legal_index_name("idx") is True
        assert not is_legal_index_name("")

    def test_is_legal_field_name(self):
        assert is_legal_field_name("vec") is True
        assert not is_legal_field_name("")

    def test_is_legal_anns_field_none(self):
        assert is_legal_anns_field(None) is True
        assert is_legal_anns_field("vec") is True
        assert is_legal_anns_field(123) is False

    def test_parser_range_date_date_obj(self):
        d = datetime.date(2024, 1, 15)
        assert parser_range_date(d) == "2024-01-15"

    def test_parser_range_date_invalid_str(self):
        with pytest.raises(ParamError):
            parser_range_date("not-a-date")

    def test_parser_range_date_invalid_type(self):
        with pytest.raises(ParamError):
            parser_range_date(12345)


class TestParamChecker:
    def test_unknown_param_raises(self):
        checker = ParamChecker()
        with pytest.raises(ParamError, match="unknown param"):
            checker.check("unknown_key", "value")

    def test_known_param_illegal_raises(self):
        checker = ParamChecker()
        with pytest.raises(ParamError):
            checker.check("limit", -1)

    def test_known_param_valid_passes(self):
        checker = ParamChecker()
        checker.check("limit", 10)


class TestCheckPassParam:
    def test_valid(self):
        check_pass_param(limit=10)

    def test_invalid_raises(self):
        with pytest.raises(ParamError):
            check_pass_param(limit=0)

    def test_unknown_raises(self):
        with pytest.raises(ParamError):
            check_pass_param(unknown_key="val")


class TestCheckIdAndData:
    def test_both_none_raises(self):
        with pytest.raises(ParamError):
            check_id_and_data(None, None)

    def test_both_provided_raises(self):
        with pytest.raises(ParamError):
            check_id_and_data([1, 2], [[0.1]])

    def test_ids_only(self):
        check_id_and_data([1, 2], None)

    def test_data_only(self):
        check_id_and_data(None, [[0.1, 0.2]])


@pytest.mark.parametrize(
    "kwargs,check",
    [
        ({ITERATOR_FIELD: True}, lambda r, kw: r is True),
        ({}, lambda r, kw: r is True and GUARANTEE_TIMESTAMP in kw),
        ({"consistency_level": "Strong"}, lambda r, kw: kw[GUARANTEE_TIMESTAMP] == 0),
        ({"consistency_level": "Bounded"}, lambda r, kw: kw[GUARANTEE_TIMESTAMP] == BOUNDED_TS),
        ({"consistency_level": "Session"}, lambda r, kw: GUARANTEE_TIMESTAMP in kw),
        (
            {"consistency_level": "Eventually"},
            lambda r, kw: kw[GUARANTEE_TIMESTAMP] == EVENTUALLY_TS,
        ),
    ],
)
def test_construct_guarantee_ts(kwargs, check):
    result = construct_guarantee_ts("col", kwargs)
    assert check(result, kwargs)


class TestMilvusClientCheck:
    def test_validate_param_none_raises(self):
        with pytest.raises(ParamError, match="missing required"):
            validate_param("foo", None, str)

    def test_validate_param_wrong_type_raises(self):
        with pytest.raises(ParamError, match="wrong type"):
            validate_param("foo", 123, str)

    def test_validate_param_correct_type_passes(self):
        validate_param("foo", "bar", str)

    def test_validate_params_dict_ok(self):
        validate_params({"a": 1, "b": 2}, int)

    def test_validate_params_wrong_type_raises(self):
        with pytest.raises(ParamError):
            validate_params({"a": "not_int"}, int)

    def test_validate_params_not_dict_raises(self):
        with pytest.raises(ParamError):
            validate_params("not_a_dict", str)

    def test_validate_noneable_param_none_ok(self):
        validate_noneable_param("foo", None, str)

    def test_validate_noneable_param_correct_type_ok(self):
        validate_noneable_param("foo", "bar", str)

    def test_validate_noneable_param_wrong_type_raises(self):
        with pytest.raises(ParamError):
            validate_noneable_param("foo", 123, str)
