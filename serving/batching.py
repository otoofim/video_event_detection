"""Async micro-batcher: coalesce concurrent single-item requests into one model call."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np


class MicroBatcher:
    """Collects items for up to ``max_wait_ms`` (or until ``max_batch``) and runs ``fn`` once.

    ``fn`` maps a stacked ``(N, ...)`` array to ``(N, ...)`` results and runs in a worker thread so
    the event loop keeps accepting frames while ONNX Runtime (which releases the GIL) computes.
    """

    def __init__(
        self, fn: Callable[[np.ndarray], np.ndarray], max_batch: int = 16, max_wait_ms: float = 3.0
    ):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.batch_sizes: list[int] = []

    async def start(self) -> None:
        self._q = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, item: np.ndarray) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        await self._q.put((item, fut))
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._q.get()
            items = [first]
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self._q.get(), remaining))
                except asyncio.TimeoutError:
                    break
            self.batch_sizes.append(len(items))
            try:
                out = await loop.run_in_executor(None, self.fn, np.stack([i for i, _ in items]))
                for (_, fut), o in zip(items, out, strict=True):
                    if not fut.done():
                        fut.set_result(o)
            except Exception as exc:  # propagate to every waiter, keep the worker alive
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)
