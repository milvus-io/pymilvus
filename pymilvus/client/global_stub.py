import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

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


def fetch_topology(global_endpoint: str, token: str) -> GlobalTopology:
    """Fetch the global cluster topology from the REST API.

    Args:
        global_endpoint: The global cluster endpoint URL
        token: Authentication token

    Returns:
        GlobalTopology object containing cluster information

    Raises:
        MilvusException: If topology cannot be fetched after retries
    """
    # Build the topology URL
    endpoint = global_endpoint.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    url = f"{endpoint}/{GLOBAL_CLUSTER_IDENTIFIER}/topology"

    headers = {"Authorization": f"Bearer {token}"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                msg = f"Topology request failed with status {response.status_code}: {response.text}"
                raise RuntimeError(msg)  # Retryable HTTP error

            result = response.json()
            if result.get("code", 0) != 0:
                raise MilvusException(message=result.get("message", "Unknown API error"))

            return _parse_topology_response(result["data"])

        except MilvusException:
            raise
        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                delay += random.uniform(0, delay * 0.1)  # noqa: S311
                logger.warning(
                    f"Topology fetch attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

    raise MilvusException(
        message=f"Failed to fetch global topology after {MAX_RETRIES} attempts: {last_error}"
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
    ):
        self._global_endpoint = global_endpoint
        self._token = token
        self._topology = topology
        self._refresh_interval = refresh_interval
        self._on_topology_change = on_topology_change

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        """Trigger an immediate topology refresh (async)."""
        threading.Thread(target=self._try_refresh, daemon=True).start()

    def _refresh_loop(self) -> None:
        """Main refresh loop running in background thread."""
        while not self._stop_event.wait(self._refresh_interval):
            self._try_refresh()

    def _try_refresh(self) -> None:
        """Attempt to refresh the topology."""
        try:
            new_topology = fetch_topology(self._global_endpoint, self._token)

            with self._lock:
                if new_topology.version > self._topology.version:
                    old_topology = self._topology
                    self._topology = new_topology
                    logger.info(
                        f"Topology updated: version {old_topology.version} -> {new_topology.version}"
                    )

                    if self._on_topology_change:
                        try:
                            self._on_topology_change(new_topology)
                        except Exception:
                            logger.warning("Topology change callback failed", exc_info=True)

        except Exception:
            logger.warning("Topology refresh failed", exc_info=True)
            # Keep using cached topology, will retry next interval
