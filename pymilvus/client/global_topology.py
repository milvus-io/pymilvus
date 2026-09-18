import logging
import random
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

import dns.exception
import dns.resolver
import requests

from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_IDENTIFIER = "global-cluster"


def is_global_endpoint(uri: str) -> bool:
    """Check if the URI points to a global cluster endpoint."""
    if not uri:
        return False
    return GLOBAL_CLUSTER_IDENTIFIER in uri.lower()


class ClusterCapability:
    """Bitset flags for cluster capabilities."""

    READABLE = 0b01  # bit 0
    WRITABLE = 0b10  # bit 1
    PRIMARY = 0b11  # read + write


@dataclass
class ClusterInfo:
    """Information about a cluster in the global topology."""

    cluster_id: str
    endpoint: str
    capability: int

    @property
    def is_primary(self) -> bool:
        """Check if this cluster is the primary (writable) cluster."""
        return (self.capability & ClusterCapability.WRITABLE) != 0


@dataclass
class GlobalTopology:
    """Global cluster topology containing all clusters."""

    version: int
    clusters: List[ClusterInfo]

    @property
    def primary(self) -> ClusterInfo:
        """Get the primary cluster from the topology."""
        for cluster in self.clusters:
            if cluster.is_primary:
                return cluster
        msg = "No primary cluster found in topology"
        raise ValueError(msg)


# Constants for retry logic
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds
REQUEST_TIMEOUT = 10  # seconds

# SRV-based topology discovery.
#
# The SDK never talks to the global endpoint directly. It derives an SRV query
# name from the endpoint hostname, resolves it to a set of ha-manager seed
# servers, and fetches topology from them. The global cluster id is NOT parsed
# here: the endpoint hostname is passed verbatim to the server via the
# ``?endpoint=`` query parameter, and the server derives the id from it (its
# first DNS label). Keeping the parse server-side lets the endpoint format
# evolve without touching every SDK.
SRV_SERVICE_PREFIX = "_grpc._tcp."
# Query parameter carrying the raw endpoint hostname to the topology API.
ENDPOINT_QUERY_PARAM = "endpoint"
# Number of seeds probed concurrently on the first fetch (nothing cached yet),
# spanning priorities so a slow or dead nearest region does not stall discovery.
# Refresh paths (a version is already cached) probe every seed instead.
DEFAULT_PROBE_COUNT = 2


def _parse_topology_response(data: dict) -> GlobalTopology:
    """Parse the topology response from the REST API."""
    clusters = [
        ClusterInfo(
            cluster_id=c["clusterId"],
            endpoint=c["endpoint"],
            capability=c["capability"],
        )
        for c in data["clusters"]
    ]
    return GlobalTopology(version=int(data["version"]), clusters=clusters)


@dataclass
class SrvTarget:
    """A single ha-manager seed server resolved from an SRV record."""

    priority: int
    weight: int
    port: int
    target: str  # ha-manager seed server hostname


def _endpoint_hostname(global_endpoint: str) -> str:
    """Extract the bare hostname from a global cluster endpoint.

    Strips scheme, credentials, port and path so the result can be used both to
    derive the SRV query name and as the verbatim ``?endpoint=`` value.
    """
    endpoint = global_endpoint.strip()
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    hostname = urlparse(endpoint).hostname
    if not hostname:
        raise MilvusException(message=f"Invalid global cluster endpoint: {global_endpoint}")
    return hostname


def _resolve_srv(hostname: str) -> List[SrvTarget]:
    """Resolve the ``_grpc._tcp.<hostname>`` SRV record set.

    Returns an empty list when the name resolves but carries no SRV records
    (NXDOMAIN / empty answer) so the caller can raise a clear "not provisioned"
    error. Transient DNS failures (timeout, no reachable nameserver) are raised.
    """
    srv_name = f"{SRV_SERVICE_PREFIX}{hostname}"
    try:
        answers = dns.resolver.resolve(srv_name, "SRV", lifetime=REQUEST_TIMEOUT)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return []
    except dns.exception.DNSException as e:
        raise MilvusException(message=f"SRV resolution failed for {srv_name}: {e}") from e

    targets = []
    for rr in answers:
        target = str(rr.target).rstrip(".")
        # RFC 2782: a target of "." means "service decidedly not available".
        # It strips down to an empty hostname, so skip it and let an empty
        # result take the "not provisioned" path like NXDOMAIN / NoAnswer.
        if not target:
            continue
        targets.append(
            SrvTarget(priority=rr.priority, weight=rr.weight, port=rr.port, target=target)
        )
    # Nearest first: lower priority wins; higher weight breaks ties.
    targets.sort(key=lambda t: (t.priority, -t.weight))
    return targets


def _weighted_choice(group: List[SrvTarget]) -> SrvTarget:
    """Pick one target from a same-priority group, weighted by SRV weight."""
    weights = [max(t.weight, 0) for t in group]
    if sum(weights) == 0:
        return random.choice(group)  # noqa: S311 - all-zero weights: pick uniformly
    return random.choices(group, weights=weights, k=1)[0]  # noqa: S311


def _pick_candidates(targets: List[SrvTarget], count: int = DEFAULT_PROBE_COUNT) -> List[SrvTarget]:
    """Select seeds to probe: one weighted pick per priority, lowest first.

    Spans priorities (so a dead nearest region does not stall discovery) and
    returns at least ``count`` seeds when that many exist.
    """
    by_priority = {}
    for t in targets:
        by_priority.setdefault(t.priority, []).append(t)

    picked: List[SrvTarget] = []
    for priority in sorted(by_priority):
        picked.append(_weighted_choice(by_priority[priority]))
        if len(picked) >= count:
            return picked

    # Few priority groups: top up from the remaining nearest targets.
    for t in targets:
        if len(picked) >= count:
            break
        if t not in picked:
            picked.append(t)
    return picked


def _probe_seed(base_url: str, hostname: str, token: str) -> GlobalTopology:
    """Fetch topology from a single seed. Raises on any failure.

    The global cluster is identified by the raw ``hostname``, passed verbatim via
    the ``?endpoint=`` query parameter; the server derives the id from it.
    """
    url = f"{base_url}/{GLOBAL_CLUSTER_IDENTIFIER}/topology"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params={ENDPOINT_QUERY_PARAM: hostname},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        msg = f"Topology request failed with status {response.status_code}: {response.text}"
        raise RuntimeError(msg)

    result = response.json()
    if result.get("code", 0) != 0:
        raise MilvusException(message=result.get("message", "Unknown API error"))
    return _parse_topology_response(result["data"])


# Every probe thread posts exactly one of these: (target, topology, error).
_ProbeResult = Tuple[SrvTarget, Optional[GlobalTopology], Optional[BaseException]]


def _probe_into_queue(
    results: "Queue[_ProbeResult]",
    target: SrvTarget,
    hostname: str,
    token: str,
) -> None:
    """Probe one seed and always post exactly one result to the queue."""
    try:
        topology = _probe_seed(f"https://{target.target}:{target.port}", hostname, token)
    except BaseException as e:
        results.put((target, None, e))
    else:
        results.put((target, topology, None))


def _watch_remaining(
    results: "Queue[_ProbeResult]",
    remaining: int,
    topology: GlobalTopology,
    on_higher_version: Callable[[GlobalTopology], None],
) -> None:
    """Drain the answers still in flight and trigger replacement on a newer one.

    Runs in a daemon thread after the first answer is handed over, so a slow
    seed does not stall the initial connection while a newer version from the
    remaining seeds can still win the replacement flow. Only results the caller
    did not consume arrive here, so each seed is logged at most once.
    """

    def watch() -> None:
        for _ in range(remaining):
            target, new_topology, error = results.get()
            if error is not None:  # one bad seed must not fail the rest
                logger.warning(f"Topology probe failed for {target.target}: {error}")
                continue
            if new_topology.version > topology.version:
                try:
                    on_higher_version(new_topology)
                except Exception:
                    logger.warning("Topology replacement callback failed", exc_info=True)
                return  # one replacement is enough; the refresher keeps polling

    threading.Thread(target=watch, daemon=True).start()


def _probe_candidates(
    candidates: List[SrvTarget],
    hostname: str,
    token: str,
    wait_all: bool,
    on_higher_version: Optional[Callable[[GlobalTopology], None]] = None,
) -> Tuple[Optional[GlobalTopology], Optional[MilvusException], Optional[BaseException]]:
    """Probe seeds concurrently; return (best topology, sticky API error, last error).

    With ``wait_all`` (a topology version is already cached locally) every seed
    is polled and the highest version wins, guarding the cache against stale
    answers. Without it (first fetch, nothing to guard) the first seed that
    answers wins so a slow seed does not stall the initial connection; the
    seeds still in flight keep being polled in the background and
    ``on_higher_version`` triggers the replacement flow when one returns a
    newer version. An API-level error (e.g. auth failure) is the same across
    seeds, so it is captured and surfaced when no seed succeeds instead of a
    generic "unreachable". The last non-API error is returned as well so the
    caller can keep the underlying cause in its final exception.

    Probes run in plain daemon threads (not a ThreadPoolExecutor, whose
    non-daemon workers are joined at interpreter exit): once the caller has
    its answer, leftover probes must never delay process shutdown.
    """
    best: Optional[GlobalTopology] = None
    api_error: Optional[MilvusException] = None
    last_error: Optional[BaseException] = None
    results: Queue[_ProbeResult] = Queue()
    for t in candidates:
        threading.Thread(
            target=_probe_into_queue, args=(results, t, hostname, token), daemon=True
        ).start()

    pending = len(candidates)
    while pending > 0:
        target, topology, error = results.get()
        pending -= 1
        if error is not None:
            if isinstance(error, MilvusException):
                api_error = error
            else:
                last_error = error
                logger.warning(f"Topology probe failed for {target.target}: {error}")
            continue
        if best is None or topology.version > best.version:
            best = topology
        if not wait_all:
            break  # First answer wins; the watcher keeps an eye on the rest.

    if not wait_all and best is not None and on_higher_version is not None and pending > 0:
        # First fetch: hand the results still in flight to a background watcher.
        _watch_remaining(results, pending, best, on_higher_version)
    return best, api_error, last_error


def _backoff_delay(attempt: int) -> float:
    """Compute the exponential backoff delay (+jitter) for the next attempt."""
    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
    return delay + random.uniform(0, delay * 0.1)  # noqa: S311


def fetch_topology(
    global_endpoint: str,
    token: str,
    cached_version: Optional[int] = None,
    on_topology_change: Optional[Callable[[GlobalTopology], None]] = None,
) -> Optional[GlobalTopology]:
    """Fetch the global cluster topology via SRV-based seed discovery.

    Resolves ``_grpc._tcp.<hostname>`` to ha-manager seeds, then concurrently
    probes the nearest ones and returns the topology with the highest version.

    Args:
        global_endpoint: The global cluster endpoint (URL or bare hostname).
        token: Authentication token.
        cached_version: Version of the topology already cached locally, if any.
            When set, every seed is polled and the highest version wins; when
            None (first fetch) the first seed that answers wins, and the rest
            keep being polled in the background.
        on_topology_change: Replacement flow invoked with a topology whose
            version is higher than the first answer (only used when
            ``cached_version`` is None).

    Returns:
        GlobalTopology object containing cluster information, or None when
        ``cached_version`` is set and no seed reports a strictly higher
        version (the cached topology is already up to date).

    Raises:
        MilvusException: If the endpoint has no SRV records, the server rejects
            the request, or DNS / seed probing fails after retries.
    """
    hostname = _endpoint_hostname(global_endpoint)
    candidates: List[SrvTarget] = []
    last_error: Optional[BaseException] = None

    for attempt in range(MAX_RETRIES):
        # Resolve inside the retry loop: DNS timeouts / unreachable nameservers
        # are transient and deserve the same backoff as an unreachable seed.
        # Only NXDOMAIN / empty answers (a deterministic "not provisioned"
        # state) fail fast below.
        try:
            targets = _resolve_srv(hostname)
        except MilvusException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = _backoff_delay(attempt)
            logger.warning(
                f"SRV resolution failed for '{hostname}' (attempt {attempt + 1}): "
                f"{e}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            continue

        if not targets:
            raise MilvusException(
                message=(
                    f"No SRV records found for global cluster endpoint "
                    f"'{hostname}' ({SRV_SERVICE_PREFIX}{hostname})"
                )
            )
        # First fetch: probe the nearest DEFAULT_PROBE_COUNT seeds so a slow
        # region does not stall the initial connection. With a cached version
        # the cache must not regress, so probe every seed and take the highest
        # version - partial replication means the nearest seeds may lag.
        candidates = _pick_candidates(
            targets, count=len(targets) if cached_version is not None else DEFAULT_PROBE_COUNT
        )

        best, api_error, probe_error = _probe_candidates(
            candidates,
            hostname,
            token,
            wait_all=cached_version is not None,
            on_higher_version=on_topology_change if cached_version is None else None,
        )
        if probe_error is not None:
            last_error = probe_error
        if best is not None:
            if cached_version is None or best.version > cached_version:
                return best
            # Every reachable seed is at or behind the cached version:
            # nothing newer on the server side, keep the cached topology.
            return None
        if api_error is not None:
            # Deterministic server-side rejection (e.g. auth): do not retry.
            raise api_error
        if attempt < MAX_RETRIES - 1:
            delay = _backoff_delay(attempt)
            logger.warning(
                f"All {len(candidates)} topology seeds unreachable "
                f"(attempt {attempt + 1}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    msg = (
        f"Failed to fetch global topology from {len(candidates)} seed(s) "
        f"for '{hostname}' after {MAX_RETRIES} attempts"
    )
    if last_error is not None:
        msg += f"; last error: {last_error}"
    raise MilvusException(message=msg)


# Default refresh interval
DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes


class TopologyRefresher:
    """Background thread that periodically refreshes the global cluster topology.

    The connection strategy is the single owner of the shared topology: the
    refresher keeps no private snapshot. ``get_current`` provides the
    authoritative copy for version comparisons, and every candidate update is
    handed to ``on_topology_change`` (the strategy's compare-and-set), whose
    return value decides whether the update was accepted.
    """

    def __init__(
        self,
        global_endpoint: str,
        token: str,
        get_current: Callable[[], Optional[GlobalTopology]],
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
        on_topology_change: Optional[Callable[[GlobalTopology], bool]] = None,
    ):
        self._global_endpoint = global_endpoint
        self._token = token
        self._get_current = get_current
        self._refresh_interval = refresh_interval
        self._on_topology_change = on_topology_change

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._refreshing = False  # Debounce flag to prevent concurrent refreshes

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_running(self) -> bool:
        """Check if the refresh thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def trigger_refresh(self) -> None:
        """Trigger an immediate topology refresh (async, debounced)."""
        with self._lock:
            if self._refreshing:
                # Already refreshing, skip to avoid duplicate requests
                return
            self._refreshing = True

        threading.Thread(target=self._try_refresh_with_cleanup, daemon=True).start()

    def _refresh_loop(self) -> None:
        """Main refresh loop running in background thread."""
        while not self._stop_event.wait(self._refresh_interval):
            self._try_refresh()

    def _try_refresh(self) -> None:
        """Attempt to refresh the topology."""
        try:
            current = self._get_current()
            if current is None:
                return
            new_topology = fetch_topology(
                self._global_endpoint, self._token, cached_version=current.version
            )

            if new_topology is not None and new_topology.version > current.version:
                accepted = False
                if self._on_topology_change:
                    try:
                        accepted = bool(self._on_topology_change(new_topology))
                    except Exception:
                        logger.warning("Topology change callback failed", exc_info=True)
                # Log only when the compare-and-set actually accepted the
                # update: a concurrent on_unavailable may have installed an
                # even newer version, and claiming v_old -> v_new then would
                # read as a rollback.
                if accepted:
                    logger.info(
                        f"Topology updated: version {current.version} -> {new_topology.version}"
                    )

        except Exception:
            logger.warning("Topology refresh failed", exc_info=True)
            # Keep using cached topology, will retry next interval

    def _try_refresh_with_cleanup(self) -> None:
        """Attempt to refresh the topology and reset the refreshing flag."""
        try:
            self._try_refresh()
        finally:
            with self._lock:
                self._refreshing = False
