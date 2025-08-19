import datetime
import logging
import re

import numpy as np
import pytest

# For tests
from pymilvus import *
from pymilvus.client import get_commit
from pymilvus.client.check import (
    check_pass_param,
    is_legal_address,
    is_legal_host,
    is_legal_port,
)
from pymilvus.client.utils import (
    hybridts_to_unixtime,
    mkts_from_datetime,
    mkts_from_hybridts,
    mkts_from_unixtime,
)

log = logging.getLogger(__name__)


class TestChecks:
    @pytest.mark.parametrize("valid_address", [
        "localhost:19530",
        "example.com:19530"
    ])
    def test_check_is_legal_address_true(self, valid_address):
        valid = is_legal_address(valid_address)
        assert valid is True

    @pytest.mark.parametrize("invalid_address", [
        "-1",
        "localhost",
        ":19530",
        "localhost:localhost",
    ])
    def test_check_is_legal_address_false(self, invalid_address):
        valid = is_legal_address(invalid_address)
        assert valid is False

    @pytest.mark.parametrize("valid_host", [
        "localhost",
        "example.com"
    ])
    def test_check_is_legal_host_true(self, valid_host):
        valid = is_legal_host(valid_host)
        assert valid is True

    @pytest.mark.parametrize("invalid_host", [
        -1,
        1.0,
        "",
        is_legal_address,
    ])
    def test_check_is_legal_host_false(self, invalid_host):
        valid = is_legal_host(invalid_host)
        assert valid is False

    @pytest.mark.parametrize("valid_port", [
        "19530",
        "222",
        123,
    ])
    def test_check_is_legal_port_true(self, valid_port):
        valid = is_legal_port(valid_port)
        assert valid is True

    @pytest.mark.parametrize("invalid_port", [
        is_legal_address,
        "abc",
        0.3,
    ])
    def test_check_is_legal_port_false(self, invalid_port):
        valid = is_legal_port(invalid_port)
        assert valid is False


class TestCheckPassParam:
    def test_check_pass_param_valid(self):
        a = [[i * j for i in range(20)] for j in range(20)]
        check_pass_param(search_data=a)

        a = np.float32([[1, 2, 3, 4], [1, 2, 3, 4]])
        check_pass_param(search_data=a)

    def test_check_param_invalid(self):
        with pytest.raises(TypeError):
            a = {[i * j for i in range(20) for j in range(20)]}
            check_pass_param(search_data=a)


class TestGenTS:
    def test_mkts1(self):
        ts = 426152581543231492
        msecs = 1000
        timestamp = hybridts_to_unixtime(ts)
        t1 = mkts_from_hybridts(ts, milliseconds=msecs)
        t2 = mkts_from_unixtime(timestamp, msecs)
        timestamp1 = hybridts_to_unixtime(t1)
        timestamp2 = hybridts_to_unixtime(t2)

        assert timestamp1 == timestamp2

        dtime = datetime.datetime.fromtimestamp(timestamp)
        t3 = mkts_from_datetime(dtime, milliseconds=msecs)
        timestamp3 = hybridts_to_unixtime(t3)
        assert timestamp1 == timestamp3

    def test_mkts2(self):
        ts = 426152581543231492
        delta = datetime.timedelta(milliseconds=1000)
        timestamp = hybridts_to_unixtime(ts)
        t1 = mkts_from_hybridts(ts, delta=delta)
        t2 = mkts_from_unixtime(timestamp, delta=delta)
        timestamp1 = hybridts_to_unixtime(t1)
        timestamp2 = hybridts_to_unixtime(t2)

        assert timestamp1 == timestamp2

        dtime = datetime.datetime.fromtimestamp(timestamp)
        t3 = mkts_from_datetime(dtime, delta=delta)
        timestamp3 = hybridts_to_unixtime(t3)
        assert timestamp1 == timestamp3


class TestGetCommit:
    def test_get_commit(self):
        s = get_commit("2.0.0rc9.dev22")
        assert s == "290d76f"

        s = get_commit("2.0.0rc8", False)
        assert s == "c9f015a04058638a28e1d2a5b265147cda0b0a23"

    def test_version_re(self):
        version_info = r'((\d+)\.(\d+)\.(\d+))((rc)(\d+))?(\.dev(\d+))?'
        p = re.compile(version_info)

        versions = [
            '2.0.0',
            '2.0.0rc3',
            '2.0.0rc4.dev8',
            '2.0.0rc4.dev22',
            '2.0.'
        ]

        for v in versions:
            rv = p.match(v)
            if rv is not None:
                assert rv.group() == v

                log.info(f"group {rv.group()}")
                log.info(f"group {rv.groups()}")
