import threading
from unittest.mock import MagicMock, patch

import dns.exception
import dns.resolver
import pytest
import requests
from pymilvus.client.global_topology import (
    GLOBAL_CLUSTER_IDENTIFIER,
    ClusterCapability,
    ClusterInfo,
    GlobalTopology,
    SrvTarget,
    TopologyRefresher,
    _endpoint_hostname,
    _pick_candidates,
    _resolve_srv,
    _weighted_choice,
    fetch_topology,
    is_global_endpoint,
)
from pymilvus.exceptions import MilvusException

# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_topology(version=1, clusters=None):
    if clusters is None:
        clusters = [
            ClusterInfo(cluster_id="in01", endpoint="https://in01.example.com", capability=3)
        ]
    return GlobalTopology(version=version, clusters=clusters)


def _mock_response(
    version="1",
    cluster_id="in01-xxx",
    endpoint="https://in01-xxx.example.com",
    capability=3,
    code=0,
    message=None,
):
    resp = MagicMock()
    resp.status_code = 200
    body = {
        "code": code,
        "data": {
            "version": version,
            "clusters": [{"clusterId": cluster_id, "endpoint": endpoint, "capability": capability}],
        },
    }
    if message is not None:
        body["message"] = message
        body.pop("data", None)
    resp.json.return_value = body
    return resp


# ── TestIsGlobalEndpoint ──────────────────────────────────────────────────────


class TestIsGlobalEndpoint:
    def test_detects_global_cluster_in_url(self):
        assert is_global_endpoint("https://glo-xxx.global-cluster.vectordb.example.com") is True

    def test_detects_global_cluster_case_insensitive(self):
        assert is_global_endpoint("https://glo-xxx.GLOBAL-CLUSTER.vectordb.example.com") is True

    def test_rejects_regular_endpoint(self):
        assert is_global_endpoint("https://in01-xxx.example.com") is False

    def test_rejects_empty_string(self):
        assert is_global_endpoint("") is False


# ── TestClusterCapability ─────────────────────────────────────────────────────


class TestClusterCapability:
    def test_primary_capability(self):
        assert ClusterCapability.PRIMARY == 0b11
        assert ClusterCapability.READABLE == 0b01
        assert ClusterCapability.WRITABLE == 0b10


# ── TestClusterInfo ───────────────────────────────────────────────────────────


class TestClusterInfo:
    def test_primary_cluster(self):
        cluster = ClusterInfo(
            cluster_id="in01-xxx", endpoint="https://in01-xxx.example.com", capability=3
        )
        assert cluster.is_primary is True

    def test_secondary_cluster(self):
        cluster = ClusterInfo(
            cluster_id="in02-xxx", endpoint="https://in02-xxx.example.com", capability=1
        )
        assert cluster.is_primary is False


# ── TestGlobalTopology ────────────────────────────────────────────────────────


class TestGlobalTopology:
    def test_finds_primary_cluster(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.example.com", capability=3
                ),
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.example.com", capability=1
                ),
            ],
        )
        primary = topology.primary
        assert primary.cluster_id == "in01-xxx"
        assert primary.is_primary is True

    def test_raises_when_no_primary(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.example.com", capability=1
                ),
            ],
        )
        with pytest.raises(ValueError, match="No primary cluster"):
            _ = topology.primary


# ── TestFetchTopology ─────────────────────────────────────────────────────────

_GLOBAL_URL = "https://glo-xxx.global-cluster.vectordb.example.com"
_GLOBAL_HOST = "glo-xxx.global-cluster.vectordb.example.com"
_TOKEN = "test-token"


class _FakeSrvRecord:
    """Mimics a dnspython SRV rdata item (target carries a trailing dot)."""

    def __init__(self, target, priority=10, weight=50, port=443):
        self.priority = priority
        self.weight = weight
        self.port = port
        self.target = f"{target}."


def _patch_srv(*records):
    """Patch DNS resolution to return the given fake SRV records."""
    return patch(
        "pymilvus.client.global_topology.dns.resolver.resolve",
        return_value=list(records),
    )


class TestEndpointHostname:
    def test_strips_scheme_port_and_path(self):
        assert _endpoint_hostname(f"https://{_GLOBAL_HOST}:443/mydb") == _GLOBAL_HOST

    def test_bare_hostname(self):
        assert _endpoint_hostname(_GLOBAL_HOST) == _GLOBAL_HOST

    def test_strips_credentials(self):
        assert _endpoint_hostname(f"https://user:pass@{_GLOBAL_HOST}") == _GLOBAL_HOST


class TestResolveSrv:
    def test_returns_targets_sorted_by_priority_then_weight(self):
        with _patch_srv(
            _FakeSrvRecord("t-us.example.com", priority=20, weight=50),
            _FakeSrvRecord("t-ap-low.example.com", priority=10, weight=10),
            _FakeSrvRecord("t-ap-high.example.com", priority=10, weight=90),
        ):
            targets = _resolve_srv(_GLOBAL_HOST)
        # priority ascending, then weight descending; trailing dot stripped
        assert [t.target for t in targets] == [
            "t-ap-high.example.com",
            "t-ap-low.example.com",
            "t-us.example.com",
        ]
        assert targets[0].port == 443

    def test_nxdomain_returns_empty(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=dns.resolver.NXDOMAIN,
        ):
            assert _resolve_srv(_GLOBAL_HOST) == []

    def test_no_answer_returns_empty(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=dns.resolver.NoAnswer,
        ):
            assert _resolve_srv(_GLOBAL_HOST) == []

    def test_transient_dns_error_raises(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=dns.exception.Timeout,
        ):
            with pytest.raises(MilvusException, match="SRV resolution failed"):
                _resolve_srv(_GLOBAL_HOST)


class TestPickCandidates:
    def test_spans_priorities(self):
        targets = [
            SrvTarget(priority=10, weight=50, port=443, target="ap1"),
            SrvTarget(priority=10, weight=50, port=443, target="ap2"),
            SrvTarget(priority=20, weight=50, port=443, target="us1"),
        ]
        picked = _pick_candidates(targets, count=2)
        assert len(picked) == 2
        assert {t.priority for t in picked} == {10, 20}  # one per priority

    def test_single_priority_group_tops_up(self):
        targets = [
            SrvTarget(priority=10, weight=50, port=443, target="ap1"),
            SrvTarget(priority=10, weight=50, port=443, target="ap2"),
        ]
        picked = _pick_candidates(targets, count=2)
        assert len(picked) == 2
        assert {t.target for t in picked} == {"ap1", "ap2"}

    def test_fewer_targets_than_count(self):
        targets = [SrvTarget(priority=10, weight=50, port=443, target="only")]
        assert _pick_candidates(targets, count=2) == targets

    def test_all_zero_weights_picks_randomly(self):
        targets = [
            SrvTarget(priority=0, weight=0, port=443, target="a"),
            SrvTarget(priority=0, weight=0, port=443, target="b"),
        ]
        with patch("pymilvus.client.global_topology.random.choice", return_value=targets[1]):
            assert _weighted_choice(targets) is targets[1]


class TestFetchTopology:
    def test_fetches_topology_successfully(self):
        mock_response = _mock_response(version="123")
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.vectordb.example.com")), patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ) as mock_get:
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert mock_get.called
            call_args = mock_get.call_args
            url = call_args[0][0]
            # connects to the SRV target, not the global endpoint
            assert "ha-mgr-ap1.global-cluster.vectordb.example.com" in url
            assert f"/{GLOBAL_CLUSTER_IDENTIFIER}/topology" in url
            assert call_args[1]["headers"]["Authorization"] == f"Bearer {_TOKEN}"
            # the raw endpoint hostname is passed verbatim; server derives the gcid
            assert call_args[1]["params"]["endpoint"] == _GLOBAL_HOST
            assert topology.version == 123
            assert topology.primary.cluster_id == "in01-xxx"

    def test_raises_when_no_srv_records(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=dns.resolver.NXDOMAIN,
        ) as mock_resolve:
            with pytest.raises(MilvusException, match="No SRV records"):
                fetch_topology(_GLOBAL_URL, _TOKEN)
            # Deterministic "not provisioned" state: fail fast, no retries.
            assert mock_resolve.call_count == 1

    def test_selects_highest_version_across_seeds(self):
        # A version is cached locally: every seed is polled, highest wins.
        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[_mock_response(version="5"), _mock_response(version="9")],
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN, cached_version=1)
            assert topology.version == 9

    def test_first_answer_wins_without_cached_version(self):
        # First fetch, nothing cached: the first seed that answers wins.
        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[_mock_response(version="5"), _mock_response(version="9")],
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert topology.version == 5

    def test_background_higher_version_triggers_replacement(self):
        # The first answer is handed over immediately, but a newer answer from
        # a remaining seed still runs the replacement flow.
        replacement = threading.Event()
        received = []

        def on_topology_change(topology):
            received.append(topology)
            replacement.set()

        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[_mock_response(version="5"), _mock_response(version="9")],
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN, on_topology_change=on_topology_change)
            assert topology.version == 5
            assert replacement.wait(timeout=2.0)
        assert received[0].version == 9

    def test_retries_when_all_seeds_unreachable_then_succeeds(self):
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[requests.exceptions.ConnectionError("boom"), _mock_response(version="7")],
        ), patch("pymilvus.client.global_topology.time.sleep"):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert topology.version == 7

    def test_raises_after_all_retries_fail(self):
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=requests.exceptions.ConnectionError("down"),
        ), patch("pymilvus.client.global_topology.time.sleep"):
            with pytest.raises(MilvusException, match="Failed to fetch global topology"):
                fetch_topology(_GLOBAL_URL, _TOKEN)

    def test_raises_api_error_without_retry(self):
        mock_response = _mock_response(code=1, message="Invalid token")
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ) as mock_get, patch("pymilvus.client.global_topology.time.sleep") as mock_sleep:
            with pytest.raises(MilvusException, match="Invalid token"):
                fetch_topology(_GLOBAL_URL, _TOKEN)
            assert mock_get.call_count == 1  # single seed, one round
            mock_sleep.assert_not_called()  # deterministic rejection, no retry

    def test_srv_resolution_failure_raises(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=dns.exception.Timeout,
        ) as mock_resolve, patch("pymilvus.client.global_topology.time.sleep"):
            with pytest.raises(MilvusException, match="SRV resolution failed"):
                fetch_topology(_GLOBAL_URL, _TOKEN)
            # Transient DNS failure is retried like an unreachable seed.
            assert mock_resolve.call_count == 3

    def test_retries_transient_dns_failure_then_succeeds(self):
        with patch(
            "pymilvus.client.global_topology.dns.resolver.resolve",
            side_effect=[
                dns.exception.Timeout,
                [_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")],
            ],
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            return_value=_mock_response(version="7"),
        ), patch(
            "pymilvus.client.global_topology.time.sleep"
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert topology.version == 7


# ── TestTopologyRefresher ─────────────────────────────────────────────────────


class TestTopologyRefresher:
    def test_starts_and_stops(self):
        topology = _make_topology()
        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.1,
        )
        refresher.start()
        assert refresher.is_running()
        refresher.stop()
        assert not refresher.is_running()

    def test_updates_topology_on_version_change(self):
        initial_topology = _make_topology(version=1)
        new_topology = _make_topology(
            version=2,
            clusters=[
                ClusterInfo(cluster_id="in02", endpoint="https://in02.example.com", capability=3)
            ],
        )

        callback_called = threading.Event()
        received_topology = []

        def on_topology_change(topo):
            received_topology.append(topo)
            callback_called.set()

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            topology=initial_topology,
            refresh_interval=0.05,
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=new_topology):
            refresher.start()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert len(received_topology) == 1
        assert received_topology[0].version == 2

    def test_does_not_update_on_same_version(self):
        topology = _make_topology(version=1)
        callback_called = []
        refresh_count = threading.Event()
        call_count = 0

        def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                refresh_count.set()
            return topology

        def on_topology_change(topo):
            callback_called.append(topo)

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=counting_fetch):
            refresher.start()
            refresh_count.wait(timeout=2.0)
            refresher.stop()

        assert len(callback_called) == 0

    def test_trigger_refresh_immediate(self):
        initial_topology = _make_topology(version=1)
        new_topology = _make_topology(
            version=2,
            clusters=[
                ClusterInfo(cluster_id="in02", endpoint="https://in02.example.com", capability=3)
            ],
        )

        callback_called = threading.Event()

        def on_topology_change(topo):
            callback_called.set()

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            topology=initial_topology,
            refresh_interval=300,  # Long interval - shouldn't trigger automatically
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=new_topology):
            refresher.start()
            refresher.trigger_refresh()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert callback_called.is_set()

    def test_continues_on_fetch_failure(self):
        topology = _make_topology(version=1)
        fetch_attempted = threading.Event()

        def failing_fetch(*args, **kwargs):
            fetch_attempted.set()
            raise Exception("Network error")

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=failing_fetch):
            refresher.start()
            fetch_attempted.wait(timeout=2.0)
            assert refresher.is_running()
            refresher.stop()

        assert refresher.get_topology().version == 1  # Still has original topology


# ── TestGlobalClusterConstant ─────────────────────────────────────────────────


class TestGlobalClusterConstant:
    def test_global_cluster_identifier_constant(self):
        assert GLOBAL_CLUSTER_IDENTIFIER == "global-cluster"
