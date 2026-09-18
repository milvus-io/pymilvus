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

    def test_dot_target_is_filtered(self):
        # RFC 2782: a target of "." means "service decidedly not available";
        # it strips to an empty hostname, so treat it as no record at all.
        rec = _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")
        rec.target = "."
        with _patch_srv(rec):
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

    def test_dot_srv_target_takes_no_records_path(self):
        # The RFC 2782 "service decidedly not available" target fails fast
        # like NXDOMAIN instead of producing InvalidURL probes and retries.
        rec = _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")
        rec.target = "."
        with _patch_srv(rec), patch("pymilvus.client.global_topology.requests.get") as mock_get:
            with pytest.raises(MilvusException, match="No SRV records"):
                fetch_topology(_GLOBAL_URL, _TOKEN)
            mock_get.assert_not_called()

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

    def test_cached_version_probes_every_seed(self):
        # Partial replication: only some seeds have the newest version, so
        # with a cached version every seed is polled and the highest wins.
        hosts = [
            "ha-mgr-ap1.global-cluster.example.com",
            "ha-mgr-ap2.global-cluster.example.com",
            "ha-mgr-us1.global-cluster.example.com",
            "ha-mgr-us2.global-cluster.example.com",
            "ha-mgr-eu1.global-cluster.example.com",
        ]
        probed = []
        probed_lock = threading.Lock()

        def fake_get(url, **kwargs):
            with probed_lock:
                probed.append(url)
            # Only the farthest seed has replicated the failover to v10.
            return _mock_response(version="10" if "eu1" in url else "9")

        with _patch_srv(*[_FakeSrvRecord(h, priority=10 + i) for i, h in enumerate(hosts)]), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=fake_get,
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN, cached_version=9)
            assert topology.version == 10
            assert len(probed) == len(hosts)

    def test_first_answer_wins_without_cached_version(self):
        # First fetch, nothing cached: the first seed that answers wins.
        # The newer answer is gated behind the return, so the completion
        # order of the two probes is deterministic.
        slow_gate = threading.Event()

        def fake_get(url, **kwargs):
            if "ha-mgr-us1" in url:
                slow_gate.wait(timeout=5.0)
                return _mock_response(version="9")
            return _mock_response(version="5")

        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=fake_get,
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert topology.version == 5
            slow_gate.set()  # let the leftover background probe finish

    def test_background_higher_version_triggers_replacement(self):
        # The first answer is handed over immediately, but a newer answer from
        # a remaining seed still runs the replacement flow. The newer answer
        # is gated until the first one is returned, so the order is fixed.
        slow_gate = threading.Event()
        replacement = threading.Event()
        received = []

        def on_topology_change(topology):
            received.append(topology)
            replacement.set()

        def fake_get(url, **kwargs):
            if "ha-mgr-us1" in url:
                slow_gate.wait(timeout=5.0)
                return _mock_response(version="9")
            return _mock_response(version="5")

        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=fake_get,
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN, on_topology_change=on_topology_change)
            assert topology.version == 5
            slow_gate.set()
            assert replacement.wait(timeout=2.0)
        assert received[0].version == 9

    def test_failed_seed_logged_once_with_target(self):
        # A seed that fails around the first answer is reported exactly once,
        # with its target name, no matter which side of the handover logs it.
        probe_failed = threading.Event()

        def counting_warning(msg, *args, **kwargs):
            if "Topology probe failed" in str(msg):
                counting_warning.count += 1
                assert "ha-mgr-ap1" in str(msg)
                probe_failed.set()

        counting_warning.count = 0

        def fake_get(url, **kwargs):
            if "ha-mgr-ap1" in url:
                raise requests.exceptions.ConnectionError("ap1 down")
            return _mock_response(version="5")

        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=fake_get,
        ), patch(
            "pymilvus.client.global_topology.logger.warning",
            side_effect=counting_warning,
        ):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN, on_topology_change=lambda t: None)
            assert topology.version == 5
            assert probe_failed.wait(timeout=2.0)
            assert counting_warning.count == 1

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

    def test_final_error_keeps_last_seed_error(self):
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=requests.exceptions.ConnectionError("boom"),
        ), patch("pymilvus.client.global_topology.time.sleep"):
            with pytest.raises(MilvusException, match="last error: boom"):
                fetch_topology(_GLOBAL_URL, _TOKEN)

    def test_retry_warning_logged_before_backoff_sleep(self):
        events = []

        def fake_sleep(_delay):
            events.append("sleep")

        def fake_warning(msg, *args, **kwargs):
            if "retrying" in str(msg):
                events.append("log")

        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=requests.exceptions.ConnectionError("down"),
        ), patch("pymilvus.client.global_topology.time.sleep", side_effect=fake_sleep), patch(
            "pymilvus.client.global_topology.logger.warning", side_effect=fake_warning
        ):
            with pytest.raises(MilvusException):
                fetch_topology(_GLOBAL_URL, _TOKEN)

        # 3 attempts, 2 backoffs; each "retrying" warning precedes its sleep.
        assert events == ["log", "sleep", "log", "sleep"]

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

    def test_returns_none_when_no_seed_beats_cached_version(self):
        # Every seed is at or behind the cached version: no update available,
        # and an answered round is not retried.
        with _patch_srv(
            _FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com", priority=10),
            _FakeSrvRecord("ha-mgr-us1.global-cluster.example.com", priority=20),
        ), patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[_mock_response(version="5"), _mock_response(version="9")],
        ) as mock_get:
            assert fetch_topology(_GLOBAL_URL, _TOKEN, cached_version=10) is None
            assert mock_get.call_count == 2  # both seeds polled, single round

    def test_returns_none_when_seed_matches_cached_version(self):
        # Strictly-higher semantics: an equal version is not an update.
        with _patch_srv(_FakeSrvRecord("ha-mgr-ap1.global-cluster.example.com")), patch(
            "pymilvus.client.global_topology.requests.get",
            return_value=_mock_response(version="9"),
        ):
            assert fetch_topology(_GLOBAL_URL, _TOKEN, cached_version=9) is None


# ── TestTopologyRefresher ─────────────────────────────────────────────────────


class TestTopologyRefresher:
    @staticmethod
    def _make_refresher(holder, **kwargs):
        """A refresher wired to a shared holder, mimicking the strategy owner."""
        return TopologyRefresher(
            global_endpoint="https://glo.global-cluster.example.com",
            token="test-token",
            get_current=lambda: holder["current"],
            **kwargs,
        )

    def test_starts_and_stops(self):
        holder = {"current": _make_topology()}
        refresher = self._make_refresher(holder, refresh_interval=0.1)
        refresher.start()
        assert refresher.is_running()
        refresher.stop()
        assert not refresher.is_running()

    def test_updates_topology_on_version_change(self):
        holder = {"current": _make_topology(version=1)}
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
            holder["current"] = topo  # the strategy's compare-and-set accepts
            callback_called.set()
            return True

        refresher = self._make_refresher(
            holder, refresh_interval=0.05, on_topology_change=on_topology_change
        )

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=new_topology):
            refresher.start()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert [t.version for t in received_topology] == [2]

    def test_does_not_update_on_same_version(self):
        holder = {"current": _make_topology(version=1)}
        callback_called = []
        refresh_count = threading.Event()
        call_count = 0

        def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                refresh_count.set()
            return holder["current"]

        refresher = self._make_refresher(
            holder, refresh_interval=0.05, on_topology_change=callback_called.append
        )

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=counting_fetch):
            refresher.start()
            refresh_count.wait(timeout=2.0)
            refresher.stop()

        assert callback_called == []

    def test_trigger_refresh_immediate(self):
        holder = {"current": _make_topology(version=1)}
        new_topology = _make_topology(
            version=2,
            clusters=[
                ClusterInfo(cluster_id="in02", endpoint="https://in02.example.com", capability=3)
            ],
        )

        callback_called = threading.Event()

        def on_topology_change(topo):
            holder["current"] = topo
            callback_called.set()
            return True

        refresher = self._make_refresher(
            holder,
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
        holder = {"current": _make_topology(version=1)}
        fetch_attempted = threading.Event()

        def failing_fetch(*args, **kwargs):
            fetch_attempted.set()
            raise Exception("Network error")

        refresher = self._make_refresher(holder, refresh_interval=0.05)

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=failing_fetch):
            refresher.start()
            fetch_attempted.wait(timeout=2.0)
            assert refresher.is_running()
            refresher.stop()

        assert holder["current"].version == 1  # Still has original topology

    def test_compares_against_shared_current_topology(self):
        # Version comparisons run against the shared copy owned by the
        # strategy; the refresher keeps no private snapshot that could diverge.
        holder = {"current": _make_topology(version=9)}
        callback_received = []
        refresher = self._make_refresher(holder, on_topology_change=callback_received.append)

        with patch(
            "pymilvus.client.global_topology.fetch_topology",
            return_value=_make_topology(version=7),
        ) as mock_fetch:
            refresher._try_refresh()

        mock_fetch.assert_called_once_with(
            "https://glo.global-cluster.example.com", "test-token", cached_version=9
        )
        assert callback_received == []

    def test_shared_path_accepts_higher_version(self):
        holder = {"current": _make_topology(version=9)}
        callback_received = []
        refresher = self._make_refresher(holder, on_topology_change=callback_received.append)

        with patch(
            "pymilvus.client.global_topology.fetch_topology",
            return_value=_make_topology(version=10),
        ):
            refresher._try_refresh()

        assert [t.version for t in callback_received] == [10]

    def test_ignores_fetch_when_no_seed_beats_cached_version(self):
        # fetch_topology returns None when every seed is behind: no update.
        holder = {"current": _make_topology(version=9)}
        callback_received = []
        refresher = self._make_refresher(holder, on_topology_change=callback_received.append)

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=None):
            refresher._try_refresh()

        assert callback_received == []
        assert holder["current"].version == 9

    def test_logs_update_only_when_callback_accepts(self):
        # The strategy's compare-and-set may reject the update when a newer
        # version landed meanwhile; logging it as an update would then read
        # like a version rollback.
        holder = {"current": _make_topology(version=7)}

        with patch(
            "pymilvus.client.global_topology.fetch_topology",
            return_value=_make_topology(version=8),
        ), patch("pymilvus.client.global_topology.logger") as mock_logger:
            rejected = self._make_refresher(holder, on_topology_change=lambda topo: False)
            rejected._try_refresh()
            mock_logger.info.assert_not_called()

            accepted = self._make_refresher(holder, on_topology_change=lambda topo: True)
            accepted._try_refresh()
            mock_logger.info.assert_called_once_with("Topology updated: version 7 -> 8")


# ── TestGlobalClusterConstant ─────────────────────────────────────────────────


class TestGlobalClusterConstant:
    def test_global_cluster_identifier_constant(self):
        assert GLOBAL_CLUSTER_IDENTIFIER == "global-cluster"
