# Global Client Design for PyMilvus

## Overview

This feature adds transparent support for Milvus global clusters (similar to Amazon Aurora Global Database). When a user connects to a global endpoint, PyMilvus automatically discovers the cluster topology and routes all operations to the primary cluster.

**Key Principles:**
- Completely transparent to users - no API changes required
- Automatic detection based on URL pattern
- All operations route to primary cluster
- Background topology refresh for resilience

## User Experience

```python
# Regular connection - works as before
client = MilvusClient(uri="https://in01-xxx.zilliz.com", token="...")

# Global connection - same API, automatic global handling
client = MilvusClient(uri="https://glo-xxx.global-cluster.vectordb.zilliz.com", token="...")
```

No new parameters, no new methods, no code changes required.

## Detection Logic

```python
def is_global_endpoint(uri: str) -> bool:
    return "global-cluster" in uri.lower()
```

## Data Structures

### Topology Response Model

```python
from dataclasses import dataclass
from typing import List

class ClusterCapability:
    READABLE = 0b01  # bit 0
    WRITABLE = 0b10  # bit 1
    PRIMARY = 0b11   # read + write

@dataclass
class ClusterInfo:
    cluster_id: str
    endpoint: str
    capability: int

    @property
    def is_primary(self) -> bool:
        return (self.capability & ClusterCapability.WRITABLE) != 0

@dataclass
class GlobalTopology:
    version: int  # parsed from string
    clusters: List[ClusterInfo]

    @property
    def primary(self) -> ClusterInfo:
        for cluster in self.clusters:
            if cluster.is_primary:
                return cluster
        raise ValueError("No primary cluster found in topology")
```

### Topology REST API

- **Endpoint:** `GET https://<global-endpoint>/global-cluster/topology`
- **Auth:** `Authorization: Bearer <token>` (reuse existing token)
- **Response:**
```json
{
    "code": 0,
    "data": {
        "version": "1",
        "clusters": [
            {
                "clusterId": "in01-xxxx",
                "endpoint": "https://in01-xxx.zilliz.com",
                "capability": 3
            }
        ]
    }
}
```

Capability values:
- `3` (0b11) = primary (read + write)
- `1` (0b01) = secondary (read only)

## Initialization Flow

```
1. User calls MilvusClient(uri="glo-xxx.global-cluster...", token="...")
2. Detect "global-cluster" in URI → use GlobalStub
3. Fetch topology from REST API (with retry + backoff)
4. Parse response, find primary cluster (capability & 0b10)
5. Connect to primary cluster's endpoint using standard gRPC stub
6. Start background topology refresh thread
7. Return initialized client
```

### Retry Strategy

```python
max_retries = 3
base_delay = 1.0  # seconds
max_delay = 10.0  # seconds

for attempt in range(max_retries):
    try:
        topology = fetch_topology(global_endpoint, token)
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise ConnectionError(f"Failed to fetch global topology: {e}")
        delay = min(base_delay * (2 ** attempt), max_delay)
        delay += random.uniform(0, delay * 0.1)  # 10% jitter
        time.sleep(delay)
```

### Error Handling

- Topology fetch fails after retries → raise `MilvusException`
- No primary cluster in topology → raise `MilvusException`
- Primary endpoint connection fails → raise standard connection error

## GlobalStub Structure

```python
@dataclass
class GlobalStub:
    global_endpoint: str
    token: str
    topology: GlobalTopology
    primary_connection: GrpcHandler
```

## Background Topology Refresh

### Refresh Triggers

1. **Fixed interval:** Every 5 minutes
2. **Event-driven:** On connection errors (e.g., gRPC unavailable, timeout)

### Background Thread Logic

```python
class TopologyRefresher:
    def __init__(self, global_stub: GlobalStub):
        self.global_stub = global_stub
        self.refresh_interval = 300  # 5 minutes
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)

    def _refresh_loop(self):
        while not self.stop_event.wait(self.refresh_interval):
            self._try_refresh()

    def _try_refresh(self):
        try:
            new_topology = fetch_topology(...)
            if new_topology.version > self.global_stub.topology.version:
                self._apply_topology(new_topology)
        except Exception as e:
            logger.warning(f"Topology refresh failed: {e}")
            # Keep using cached topology, will retry next interval

    def trigger_refresh(self):
        """Called on connection errors - immediate refresh"""
        threading.Thread(target=self._try_refresh, daemon=True).start()
```

### Thread Safety

- Use `threading.Lock` when updating topology and primary connection
- Swap connection atomically: create new → update reference → close old

### On Topology Refresh (version changed)

- Reconnect to new primary if primary endpoint changed
- Close old connection after new one is established

## Operation Routing

All operations (read and write) go through the primary connection. The GlobalStub acts as a transparent wrapper.

### Integration Point

```python
class GrpcHandler:
    def __init__(self, uri, ...):
        if is_global_endpoint(uri):
            self._global_stub = GlobalStub(uri, token)
            self._channel = self._global_stub.primary_connection._channel
            self._stub = self._global_stub.primary_connection._stub
        else:
            # existing logic - direct connection
            self._channel = create_channel(uri, ...)
            self._stub = create_stub(self._channel)
            self._global_stub = None
```

### Error Handling with Refresh Trigger

```python
def _execute_request(self, request_func):
    try:
        return request_func()
    except grpc.RpcError as e:
        if self._global_stub and self._is_connection_error(e):
            self._global_stub.trigger_refresh()
        raise
```

### Connection Errors that Trigger Refresh

- `UNAVAILABLE` - server unreachable
- `DEADLINE_EXCEEDED` - timeout (optional)

## Implementation Plan

### Files to Modify/Create

| File | Change |
|------|--------|
| `pymilvus/client/grpc_handler.py` | Add global endpoint detection, integrate GlobalStub |
| `pymilvus/client/global_stub.py` | **New file** - GlobalStub, TopologyRefresher, data models |
| `pymilvus/exceptions.py` | Add `GlobalClusterError` if needed (or reuse `MilvusException`) |

### Dependencies

- `requests` or `urllib3` for REST API calls (check if already in dependencies)
- Standard library: `threading`, `dataclasses`, `time`, `random`

### Test Coverage (>90%)

- Unit tests for `is_global_endpoint()` detection
- Unit tests for topology parsing
- Unit tests for retry logic with mocked failures
- Unit tests for background refresh thread
- Integration tests with mocked global endpoint

### No Changes Required

- User-facing API (`MilvusClient`)
- Existing gRPC operations
- Authentication flow

## Scope Limitations (v1)

- **No failover support** - If primary is unavailable, operations fail
- **No read routing to secondaries** - All operations go to primary
- **No user-configurable refresh interval** - Fixed at 5 minutes
