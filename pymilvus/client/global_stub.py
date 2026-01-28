import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import requests

from pymilvus.exceptions import MilvusException

if TYPE_CHECKING:
    from pymilvus.client.grpc_handler import GrpcHandler

logger = logging.getLogger(__name__)


def is_global_endpoint(uri: str) -> bool:
    """Check if the URI points to a global cluster endpoint."""
    if not uri:
        return False
    return "global-cluster" in uri.lower()


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
        raise ValueError("No primary cluster found in topology")
