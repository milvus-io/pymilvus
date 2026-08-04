from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from pymilvus.client.constants import MAX_BATCH_SIZE

from .query_iterator import _QueryIteratorBase

if TYPE_CHECKING:
    from pymilvus.client.call_context import CallContext

log = logging.getLogger(__name__)


class AsyncQueryIteratorHandler(Protocol):
    async def describe_collection(
        self, collection_name: str, **kwargs: Any
    ) -> Mapping[str, Any]: ...

    async def query(self, collection_name: str, **kwargs: Any) -> Any: ...


class AsyncQueryIterator(_QueryIteratorBase):
    """Non-blocking counterpart of :class:`QueryIterator`.

    ``__init__`` cannot await, so setup RPCs (describe collection, mvccTs probe and the
    optional offset seek) run in :meth:`create`. Build instances through that classmethod,
    or through :meth:`AsyncMilvusClient.query_iterator`, never by calling the constructor
    directly.

    A single iterator must not be advanced concurrently: calling ``await it.next()``
    from two tasks at once corrupts the primary-key cursor, because the state mutation
    straddles the await. (The sync iterator has the same hazard across threads.)

    When a checkpoint file is configured (the ``iterator_cp_file`` option), every batch
    write goes through :meth:`_finalize_batch`, which writes and flushes one cursor line
    to that file synchronously, on the event-loop thread; the same is true of the file
    open/read done by :meth:`create` and the close/unlink done by :meth:`close`. A slow
    or network-mounted checkpoint path therefore blocks the loop and stalls unrelated
    coroutines. Leave checkpointing off, or point it at local disk, on latency-sensitive
    loops.

    Examples:
        >>> it = await client.query_iterator(collection_name="c", filter="pk > 0")
        >>> try:
        ...     async for batch in it:
        ...         handle(batch)
        ... finally:
        ...     await it.close()
    """

    @classmethod
    async def create(
        cls,
        handler: AsyncQueryIteratorHandler,
        context: CallContext | None,
        collection_name: str,
        batch_size: int | None = 1000,
        limit: int | None = -1,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        rpc_options: Mapping[str, Any] | None = None,
    ) -> AsyncQueryIterator:
        iterator = cls(
            handler=handler,
            context=context,
            collection_name=collection_name,
            batch_size=batch_size,
            limit=limit,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            schema=schema,
            timeout=timeout,
            rpc_options=rpc_options,
        )
        await iterator._setup()
        return iterator

    async def _setup(self) -> None:
        self._apply_collection_id(
            await self._handler.describe_collection(
                self._collection_name, **self._describe_kwargs()
            )
        )
        if self._prepare_ts_cp():
            self._consume_ts_response(await self._handler.query(**self._ts_query_kwargs()))
            self._save_mvcc_ts_if_needed()
        await self._seek_to_offset()

    async def _seek_to_offset(self) -> None:
        offset = self._seek_offset_start()
        if offset <= 0:
            return
        start_time = time.time()
        while offset > 0:
            batch_size = min(MAX_BATCH_SIZE, offset)
            query_kwargs = self._seek_query_kwargs(batch_size)
            seeked_count = self._consume_seek_batch(await self._handler.query(**query_kwargs))
            log.debug(
                f"seeked offset, seek_expr:{query_kwargs['expr']} "
                f"batch_size:{batch_size} seeked_count:{seeked_count}"
            )
            if seeked_count == 0:
                log.info("seek offset has drained all matched results for query iterator, break")
                break
            offset -= seeked_count
        self._finish_seek(offset, start_time)

    async def next(self) -> list:
        ret = self._take_from_cache()
        if ret is None:
            ret = self._consume_next_response(
                await self._handler.query(**self._next_query_kwargs())
            )
        return self._finalize_batch(ret)

    async def close(self) -> None:
        self._close_common()

    def __aiter__(self) -> AsyncQueryIterator:
        return self

    async def __anext__(self) -> list:
        batch = await self.next()
        if not batch:
            raise StopAsyncIteration
        return batch
