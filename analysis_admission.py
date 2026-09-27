"""Bounded, process-local admission for the existing synchronous analysis API."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager


class AdmissionFull(Exception):
    pass


class AdmissionTimeout(Exception):
    pass


class AnalysisAdmission:
    def __init__(self, workers: int, queue_size: int, wait_seconds: float):
        self.workers = workers
        self.queue_size = queue_size
        self.wait_seconds = wait_seconds
        self.semaphore = asyncio.Semaphore(workers)
        self.admitted = self.active = self.waiting = 0
        self.peak_active = self.peak_admitted = 0
        self.completed = self.failed = self.rejected = self.timed_out = 0

    @asynccontextmanager
    async def reserve(self):
        # Called before reading multipart bodies. No await between check/increment.
        if self.admitted >= self.workers + self.queue_size:
            self.rejected += 1
            raise AdmissionFull()
        self.admitted += 1
        self.peak_admitted = max(self.peak_admitted, self.admitted)
        try:
            yield
        finally:
            self.admitted -= 1

    @asynccontextmanager
    async def worker(self):
        started = time.perf_counter()
        self.waiting += 1
        try:
            await asyncio.wait_for(self.semaphore.acquire(), self.wait_seconds)
        except TimeoutError as exc:
            self.timed_out += 1
            raise AdmissionTimeout() from exc
        finally:
            self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            yield (time.perf_counter() - started) * 1000
        finally:
            self.active -= 1
            self.semaphore.release()

    def snapshot(self):
        return {name: getattr(self, name) for name in (
            'workers', 'queue_size', 'wait_seconds', 'admitted', 'active',
            'waiting', 'peak_active', 'peak_admitted', 'completed', 'failed',
            'rejected', 'timed_out',
        )}


async def finish_thread_on_cancel(awaitable):
    """Python inference cannot safely be killed; retain slot/files until it ends."""
    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Keep the thread accounted for even if the ASGI task is cancelled.
        # Repeated cancellation must not release its slot or remove its input.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve an abandoned failure without masking cancellation.
        raise
