"""Small in-process fixed-window limiter for a single service replica."""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic


class RateLimiter:
    def __init__(self, *, window_seconds: float = 60.0) -> None:
        self._window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, *, limit: int) -> bool:
        now = monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True
