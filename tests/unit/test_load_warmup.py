import pytest
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError


def test_load_sync_warmup_preserves_other_options():
    request = Prepare.load_collection(
        "warmup_test", replica_number=2, warmup="sync", load_fields=["id"], priority="HIGH"
    )
    assert request.load_params["warmup"] == "sync"
    assert request.load_params["load_priority"] == "HIGH"
    assert request.replica_number == 2
    assert list(request.load_fields) == ["id"]


def test_load_without_warmup_keeps_existing_semantics():
    assert "warmup" not in Prepare.load_collection("warmup_test").load_params
    assert Prepare.load_collection("warmup_test", refresh=True).refresh


@pytest.mark.parametrize("warmup", ["disable", "async", "", None, True, 1])
def test_load_rejects_unsupported_warmup(warmup):
    with pytest.raises(ParamError, match="only supports sync"):
        Prepare.load_collection("warmup_test", warmup=warmup)


@pytest.mark.parametrize("refresh_key", ["refresh", "_refresh"])
def test_load_sync_warmup_rejects_refresh(refresh_key):
    with pytest.raises(ParamError, match="cannot be used with refresh"):
        Prepare.load_collection("warmup_test", warmup="sync", **{refresh_key: True})
