from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event
from typing import Callable, Iterable


def run_bounded(entries: Iterable, action: Callable, concurrency: int, stop: Event) -> None:
    """Keep at most concurrency futures and wait for running workers before cleanup."""
    iterator = iter(entries)
    active = set()
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        exhausted = False
        while not exhausted or active:
            while not exhausted and len(active) < concurrency:
                entry = next(iterator, None)
                if entry is None:
                    exhausted = True
                else:
                    active.add(executor.submit(action, entry))
            if active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                active.difference_update(done)
    finally:
        stop.set()
        for future in active:
            future.cancel()
        executor.shutdown(wait=True)
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
