import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
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
# Minimum number of seeds to probe concurrently, spanning priorities so a slow
# or dead nearest region does not stall discovery.
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

    targets = [
        SrvTarget(
            priority=rr.priority,
            weight=rr.weight,
            port=rr.port,
            target=str(rr.target).rstrip("."),
        )
        for rr in answers
    ]
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


def _watch_remaining(
    executor: ThreadPoolExecutor,
    futures: Dict[Future, SrvTarget],
    answered: Future,
    topology: GlobalTopology,
    on_higher_version: Callable[[GlobalTopology], None],
) -> None:
    """Watch the seeds still in flight and trigger replacement on newer answers.

    Runs in a daemon thread after the first answer is handed over, so a slow
    seed does not stall the initial connection while a newer version from the
    remaining seeds can still win the replacement flow.
    """

    def watch() -> None:
        try:
            for future in as_completed(futures):
                if future is answered:
                    continue
                try:
                    new_topology = future.result()
                except Exception as e:  # one bad seed must not fail the rest
                    logger.warning(f"Topology probe failed: {e}")
                    continue
                if new_topology.version > topology.version:
                    try:
                        on_higher_version(new_topology)
                    except Exception:
                        logger.warning("Topology replacement callback failed", exc_info=True)
                    return  # one replacement is enough; the refresher keeps polling
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    threading.Thread(target=watch, daemon=True).start()


def _probe_candidates(
    candidates: List[SrvTarget],
    hostname: str,
    token: str,
    wait_all: bool,
    on_higher_version: Optional[Callable[[GlobalTopology], None]] = None,
) -> Tuple[Optional[GlobalTopology], Optional[MilvusException]]:
    """Probe seeds concurrently; return (best topology, sticky API error).

    With ``wait_all`` (a topology version is already cached locally) every seed
    is polled and the highest version wins, guarding the cache against stale
    answers. Without it (first fetch, nothing to guard) the first seed that
    answers wins so a slow seed does not stall the initial connection; the
    seeds still in flight keep being polled in the background and
    ``on_higher_version`` triggers the replacement flow when one returns a
    newer version. An API-level error (e.g. auth failure) is the same across
    seeds, so it is captured and surfaced when no seed succeeds instead of a
    generic "unreachable".
    """
    best: Optional[GlobalTopology] = None
    api_error: Optional[MilvusException] = None
    executor = ThreadPoolExecutor(max_workers=len(candidates))
    futures = {
        executor.submit(_probe_seed, f"https://{t.target}:{t.port}", hostname, token): t
        for t in candidates
    }
    answered: Optional[Future] = None
    try:
        for future in as_completed(futures):
            target = futures[future]
            try:
                topology = future.result()
            except MilvusException as e:
                api_error = e
                continue
            except Exception as e:  # one bad seed must not fail the rest
                logger.warning(f"Topology probe failed for {target.target}: {e}")
                continue
            if best is None or topology.version > best.version:
                best = topology
            if not wait_all:
                answered = future
                break
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    if wait_all:
        # Every seed was polled; the highest version seen wins.
        executor.shutdown(wait=False, cancel_futures=True)
    elif best is not None and on_higher_version is not None and answered is not None:
        # First fetch: hand the seeds still in flight to a background watcher.
        _watch_remaining(executor, futures, answered, best, on_higher_version)
    else:
        executor.shutdown(wait=False, cancel_futures=True)
    return best, api_error


def _backoff_sleep(attempt: int) -> float:
    """Sleep with exponential backoff (+jitter) before the next attempt.

    Returns the computed delay so callers can include it in log messages.
    """
    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
    delay += random.uniform(0, delay * 0.1)  # noqa: S311
    time.sleep(delay)
    return delay


def fetch_topology(
    global_endpoint: str,
    token: str,
    cached_version: Optional[int] = None,
    on_topology_change: Optional[Callable[[GlobalTopology], None]] = None,
) -> GlobalTopology:
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
        GlobalTopology object containing cluster information.

    Raises:
        MilvusException: If the endpoint has no SRV records, the server rejects
            the request, or DNS / seed probing fails after retries.
    """
    hostname = _endpoint_hostname(global_endpoint)
    candidates: List[SrvTarget] = []

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
            delay = _backoff_sleep(attempt)
            logger.warning(
                f"SRV resolution failed for '{hostname}' (attempt {attempt + 1}): "
                f"{e}; retrying in {delay:.1f}s"
            )
            continue

        if not targets:
            raise MilvusException(
                message=(
                    f"No SRV records found for global cluster endpoint "
                    f"'{hostname}' ({SRV_SERVICE_PREFIX}{hostname})"
                )
            )
        candidates = _pick_candidates(targets)

        best, api_error = _probe_candidates(
            candidates,
            hostname,
            token,
            wait_all=cached_version is not None,
            on_higher_version=on_topology_change if cached_version is None else None,
        )
        if best is not None:
            return best
        if api_error is not None:
            # Deterministic server-side rejection (e.g. auth): do not retry.
            raise api_error
        if attempt < MAX_RETRIES - 1:
            delay = _backoff_sleep(attempt)
            logger.warning(
                f"All {len(candidates)} topology seeds unreachable "
                f"(attempt {attempt + 1}); retrying in {delay:.1f}s"
            )

    raise MilvusException(
        message=(
            f"Failed to fetch global topology from {len(candidates)} seed(s) "
            f"for '{hostname}' after {MAX_RETRIES} attempts"
        )
    )


# Default refresh interval
DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes


class TopologyRefresher:
    """Background thread that periodically refreshes the global cluster topology."""

    def __init__(
        self,
        global_endpoint: str,
        token: str,
        topology: GlobalTopology,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
        on_topology_change: Optional[Callable] = None,
        get_current: Optional[Callable[[], Optional[GlobalTopology]]] = None,
    ):
        self._global_endpoint = global_endpoint
        self._token = token
        self._topology = topology
        self._refresh_interval = refresh_interval
        self._on_topology_change = on_topology_change
        # Optional provider of the authoritative current topology (the shared
        # copy owned by the connection strategy). When set, version comparisons
        # and cached_version derive from it so the refresher's private snapshot
        # cannot diverge and trigger a version rollback.
        self._get_current = get_current

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

    def get_topology(self) -> GlobalTopology:
        """Get the current topology (thread-safe)."""
        with self._lock:
            return self._topology

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

    def _current_topology(self) -> GlobalTopology:
        """The authoritative current topology for version comparisons.

        Uses the provider's shared copy when set, so refreshes never regress
        the version the connection strategy is actually routing on.
        """
        if self._get_current is not None:
            current = self._get_current()
            if current is not None:
                return current
        with self._lock:
            return self._topology

    def _try_refresh(self) -> None:
        """Attempt to refresh the topology."""
        try:
            current = self._current_topology()
            new_topology = fetch_topology(
                self._global_endpoint, self._token, cached_version=current.version
            )

            if new_topology.version > current.version:
                logger.info(
                    f"Topology updated: version {current.version} -> {new_topology.version}"
                )
                if self._get_current is None:
                    with self._lock:
                        self._topology = new_topology

                if self._on_topology_change:
                    try:
                        self._on_topology_change(new_topology)
                    except Exception:
                        logger.warning("Topology change callback failed", exc_info=True)

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
