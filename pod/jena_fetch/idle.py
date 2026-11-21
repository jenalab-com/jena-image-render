"""Decide when the pod should terminate itself: idle too long or past the hard limit."""

from __future__ import annotations

import threading
import time
from typing import Callable


class IdleWatch:
    def __init__(
        self,
        idle_seconds: float,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds
        self.clock = clock
        self.started = clock()
        self.last_active = self.started
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self.last_active = self.clock()

    def reason(self, busy: bool) -> str | None:
        """Why the pod should stop now, or None to keep running."""
        now = self.clock()
        if self.max_seconds and now - self.started > self.max_seconds:
            return f"max lifetime {self.max_seconds:.0f}s reached"
        if busy:
            self.touch()
            return None
        with self._lock:
            quiet = now - self.last_active
        if self.idle_seconds and quiet > self.idle_seconds:
            return f"idle for {quiet:.0f}s"
        return None


def run_watch(
    watch: IdleWatch,
    busy: Callable[[], bool],
    terminate: Callable[[str], bool],
    sleep: Callable[[float], None] = time.sleep,
    interval: float = 30.0,
) -> None:
    """Poll until termination succeeds; failures never stop the watcher."""
    while True:
        sleep(interval)
        try:
            working = busy()
        except Exception:  # an unknown state must not kill a running render
            working = True
        reason = watch.reason(busy=working)
        if not reason:
            continue
        try:
            if terminate(reason):
                return
        except Exception:  # retry on the next tick
            pass
        sleep(60.0)
