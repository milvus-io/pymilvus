# Connection Manager Design & Implementation Plan

- **Created:** 2026-02-03
- **Updated:** 2026-02-13
- **Author(s):** @XuanYang-cn

## Overview

A new ConnectionManager component that replaces the `connections` singleton for MilvusClient.

**Key Principles:**
- MilvusClient only (ORM continues using `connections` singleton until deprecated)
- Bypass `connections` singleton entirely
- Strategy pattern for regular vs global endpoints
- Shared recovery logic on UNAVAILABLE errors

**Tech Stack:** Python 3.8+, grpc, threading (sync), asyncio (async), dataclasses, weakref

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MilvusClient                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ConnectionManager                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Shared Logic                            │ │
│  │  - Registry (deduplication by address|token)              │ │
│  │  - Health check (gRPC state + GetVersion)                  │ │
│  │  - Recovery on UNAVAILABLE (reset connection)              │ │
│  │  - Lifecycle (create, release, close)                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   RegularStrategy    │    │      GlobalStrategy          │   │
│  │  - Direct creation   │    │  - Topology discovery        │   │
│  │                      │    │  - Primary routing           │   │
│  │                      │    │  - Background refresh        │   │
│  └──────────┬───────────┘    └──────────────┬───────────────┘   │
└─────────────┼───────────────────────────────┼───────────────────┘
              │         creates               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GrpcHandler                               │
│              Pure gRPC operations (simplified)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Sync/Async Split

No shared base class — sync and async managers are independent classes with
the same interface. Strategies share topology logic via `_GlobalStrategyMixin`.

```
┌───────────────────┐  ┌────────────────────────┐
│ ConnectionManager │  │ AsyncConnectionManager │
│ - threading.RLock │  │ - asyncio.Lock         │
│ - sync health     │  │ - async health         │
│ - sync recovery   │  │ - async recovery       │
└───────────────────┘  └────────────────────────┘

┌───────────────────────────────────────────┐
│          _GlobalStrategyMixin             │
│  - Shared topology fetch & refresh logic  │
│  - on_unavailable (sync, returns bool)    │
└───────────────┬───────────────────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌────────────────┐  ┌─────────────────────┐
│ GlobalStrategy │  │ AsyncGlobalStrategy │
└────────────────┘  └─────────────────────┘
```

## URI Parsing

Extract connection parameters from URI (simpler than `connections.connect`):

```
https://{user:pass|token}@host:19530/mydb
  │      │    │           │    │     │
  │      │    │           │    │     └── db_name (optional, from path)
  │      │    │           │    └── port (default: 19530)
  │      │    │           └── host
  │      │    └── password (optional, extracted to token)
  │      └── user (optional, extracted to token)
  └── scheme (validated: http/https/tcp; also unix: and .db suffix)
```

### Special URI Forms

| Form | Behavior |
|------|----------|
| `.db` suffix (e.g. `./local.db`) | Milvus-lite: validates parent dir, starts local server via `server_manager_instance`, rewrites URI to local gRPC endpoint |
| `unix:` scheme (e.g. `unix:/var/run/milvus.sock`) | Raw URI passed as `address` (no parsing) |
| `https://` scheme | Auto-sets `secure=True` in handler kwargs |

### ConnectionConfig.from_uri(uri, ...) -> ConnectionConfig

| URI Component | Maps To | Example |
|---------------|---------|---------|
| `user:pass@` | `token` | `user:pass@` → `token="user:pass"` |
| `host:port` | `address` | `host:19530` |
| `/path` | `db_name` | `/mydb` → `db_name="mydb"` |

### Examples

| URI | Result |
|-----|--------|
| `http://localhost:19530` | `address=localhost:19530, token=""` |
| `https://host:19530/mydb` | `address=host:19530, db_name=mydb, secure=True` |
| `https://user:pass@host:19530` | `address=host:19530, token=user:pass, secure=True` |
| `./local.db` | milvus-lite: starts local server, rewrites to gRPC URI |
| `unix:/var/run/milvus.sock` | `address=unix:/var/run/milvus.sock` (raw pass-through) |

### Priority

If both URI and explicit parameter provided, explicit wins:

```python
MilvusClient(uri="https://host/db1", db_name="db2")  # db_name = "db2"
MilvusClient(uri="https://user:pass@host", token="other")  # token = "other"
```

## API Reference

### ConnectionConfig

| Field     | Type | Description |
|-----------|------|-------------|
| `uri`     | `str` | Server URI (raw, before parsing) |
| `address` | `str` | Parsed `host:port` |
| `token`   | `str` | Auth token (from param or URI) |
| `db_name` | `str` | Database name (from param or URI path) |
| `key`     | `property` | `f"{address}\|{token}"` for deduplication |
| `is_global` | `property` | `True` if URI contains `global-cluster` |

### ManagedConnection

| Field | Type | Description |
|-------|------|-------------|
| `handler` | `GrpcHandler` | The wrapped handler |
| `config` | `ConnectionConfig` | Original config |
| `strategy` | `ConnectionStrategy` | Regular or Global |
| `created_at` | `float` | Creation timestamp |
| `last_used_at` | `float` | Last access timestamp |
| `clients` | `WeakSet` | Referencing clients |

### ConnectionManager API

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_instance` | `() -> ConnectionManager` | Get singleton (sync: process-wide; async: per-event-loop via `AsyncConnectionManager`) |
| `get_or_create` | `(config, dedicated=False, client=None, timeout=None) -> GrpcHandler` | Get/create handler |
| `release` | `(handler, client=None) -> None` | Release connection |
| `close_all` | `() -> None` | Close all connections |
| `handle_error` | `(handler, error) -> bool` | Handle UNAVAILABLE / REPLICATE_VIOLATION. Returns True if handled (caller should retry). |
| `get_stats` | `() -> Dict` | Pool statistics |
| `_check_health` | `(managed) -> bool` | gRPC state + GetVersion (private) |
| `_recover` | `(managed) -> None` | Reset connection (private) |

### ConnectionStrategy API (ABC)

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_handler` | `(config) -> GrpcHandler` | Create new handler |
| `on_unavailable` | `(managed) -> bool` | Return True if recovery should be triggered |
| `close` | `(managed) -> None` | Cleanup resources (sync) |
| `close_async` | `(managed) -> None` | Cleanup resources (async, default delegates to close) |

### RegularStrategy

| Method | Behavior |
|--------|----------|
| `create_handler` | Direct `GrpcHandler(uri, token, ...)` |
| `on_unavailable` | Returns `True` (always recover) |
| `close` | `handler.close()` |

### GlobalStrategy (extends _GlobalStrategyMixin)

| Method | Behavior |
|--------|----------|
| `create_handler` | Fetch topology → create handler to primary |
| `on_unavailable` | Fetch fresh topology, return `True` only if primary changed |
| `close` | Stop refresher, close handler |

## Recovery Flow (Shared Logic)

Both strategies share the same recovery logic on UNAVAILABLE:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ConnectionManager.handle_error()              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Is UNAVAILABLE?   │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │ Yes                         │ No
               ▼                             ▼
    ┌─────────────────────────┐        ┌───────────┐
    │ Release lock            │        │ No action │
    │ strategy.on_unavailable │        └───────────┘
    │ (may do network I/O)    │
    └──────────┬──────────────┘
               │ returns bool
    ┌──────────┴──────────────────────────────┐
    │ True                                    │ False
    ▼                                         ▼
┌──────────────────────────┐         ┌───────────┐
│ Re-acquire lock          │         │ No action │
│ Verify handler unchanged │         └───────────┘
│ _recover(managed)        │
└──────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 Shared: ConnectionManager._recover()             │
│  1. Close old handler via strategy.close()                       │
│  2. Create new handler via strategy.create_handler()             │
│  3. Register error callback on new handler                       │
│  4. Wait for channel ready                                       │
│  5. Update ManagedConnection.handler                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** `_recover()` is shared logic in ConnectionManager. Strategies just decide WHEN to call it via the `on_unavailable() -> bool` return value:
- **RegularStrategy**: Always returns `True` → always recover
- **GlobalStrategy**: Fetches fresh topology, returns `True` only if primary endpoint changed

**Split-lock pattern (sync):** `handle_error()` releases the lock before calling
`on_unavailable()` (which may do network I/O for GlobalStrategy). It re-acquires
the lock and re-verifies the handler hasn't been swapped by another thread before
calling `_recover()`.

**Async locking:** `AsyncConnectionManager.handle_error()` holds the `asyncio.Lock`
throughout the entire operation and runs `on_unavailable()` via `run_in_executor`
to avoid blocking the event loop. No split-lock is needed because `asyncio.Lock`
queues coroutines (doesn't block threads), and holding the lock serializes recovery
so double-recovery cannot happen.

## Health Check (On Checkout)

Runs if connection idle > 30 seconds:

1. Check `channel.check_connectivity_state()` != SHUTDOWN
2. Call `handler.get_server_version(timeout=5.0)`
3. If unhealthy → `_recover()` → return new handler

## Error Handling

| Error | Action |
|-------|--------|
| `UNAVAILABLE` | Trigger recovery via strategy |
| `STREAMING_CODE_REPLICATE_VIOLATION` (MilvusException) | Trigger recovery via strategy (same as UNAVAILABLE) |
| Other errors | No action |

### Error Callback Chain

The retry decorator (`@retry_on_rpc_failure`) notifies the connection manager of
retryable RPC errors via a callback registered on each handler.

**Sync chain** (ConnectionManager):
```
sync retry_decorator → handler._on_rpc_error(e)  [sync lambda]
    → ConnectionManager.handle_error()  [sync]
        → _recover()  [sync]
            → handler._wait_for_channel_ready()
```

**Async chain** (AsyncConnectionManager):
```
async retry_decorator → await handler._on_rpc_error(e)  [async def]
    → await AsyncConnectionManager.handle_error()  [async, uses asyncio.Lock]
        → await _recover()  [async]
            → await handler.ensure_channel_ready()
```

The async chain is fully async end-to-end, which means recovery can `await
ensure_channel_ready()` on the new handler. This guarantees the handler is ready
before it's stored in the managed connection, so individual RPC methods in
`AsyncGrpcHandler` do **not** need to call `ensure_channel_ready()` themselves.

Channel readiness is established at two points only:
1. **Creation** — `_create_shared` / `_create_dedicated` await `ensure_channel_ready()`
2. **Recovery** — `_recover` awaits `ensure_channel_ready()` on the replacement handler

### Legacy ORM Async Path

The legacy ORM `connections.connect(_async=True)` creates `AsyncGrpcHandler`
directly, bypassing `AsyncConnectionManager`. This path does not call
`ensure_channel_ready()` and is considered **unsupported internal API** (note the
`_async` underscore prefix). It will be removed when the ORM layer is deprecated.

New async code should use `AsyncMilvusClient` which uses `AsyncConnectionManager`.

## Client Integration

### MilvusClient

```python
class MilvusClient:
    def __init__(self, uri, token, *, dedicated=False, **kwargs):
        self._manager = ConnectionManager.get_instance()
        self._handler = self._manager.get_or_create(config, dedicated, client=self)

    def close(self):
        self._manager.release(self._handler, client=self)
```

### AsyncMilvusClient

`__init__` stores config only (no connection). Connection is deferred to first use.

```python
class AsyncMilvusClient:
    def __init__(self, uri, token, *, dedicated=False, **kwargs):
        self._config = ConnectionConfig.from_uri(uri, token=token, **kwargs)
        self._manager = None
        self._handler = None

    async def _connect(self):
        """Establish the async connection (idempotent)."""
        if self._handler is not None:
            return
        self._manager = AsyncConnectionManager.get_instance()
        self._handler = await self._manager.get_or_create(
            self._config, dedicated, client=self
        )

    async def __aenter__(self):
        await self._connect()
        return self

    async def _get_connection(self):
        """Return handler, auto-connecting if needed."""
        if self._handler is None:
            await self._connect()
        return self._handler

    async def close(self):
        if self._manager and self._handler:
            await self._manager.release(self._handler, client=self)
```

## File Structure

```
pymilvus/client/
├── connection_manager.py
│   ├── ConnectionConfig
│   ├── ManagedConnection
│   ├── ConnectionStrategy (ABC)
│   ├── RegularStrategy
│   ├── _GlobalStrategyMixin
│   ├── GlobalStrategy
│   ├── ConnectionManager (sync)
│   ├── AsyncRegularStrategy
│   ├── AsyncGlobalStrategy
│   └── AsyncConnectionManager
│
├── global_topology.py         # Topology helpers (fetch, refresh, data classes)
├── grpc_handler.py            # Simplified: no global detection
└── async_grpc_handler.py
```

## Migration from PR #3251

| Before (PR #3251) | After |
|-------------------|-------|
| GlobalStub inside GrpcHandler | GlobalStrategy inside ConnectionManager |
| GrpcHandler detects global URI | ConnectionManager detects global URI |
| Separate error handling | Unified via strategy + shared _recover() |
| GlobalStub class in global_stub.py | Removed (superseded by GlobalStrategy in global_topology.py) |

## Testing

- Singleton behavior
- Shared mode deduplication
- Dedicated mode isolation
- Health check on idle connections
- UNAVAILABLE recovery (both strategies)
- Global topology refresh
- Thread safety (sync) / Concurrency (async)
