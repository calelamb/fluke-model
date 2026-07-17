"""Cooperative cancellation and deadlines for blocking model operations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic


class OperationCancelledError(RuntimeError):
    pass


class OperationSupersededError(OperationCancelledError):
    pass


@dataclass(frozen=True)
class OperationDeadline:
    expires_at: float
    cancelled: Event

    @classmethod
    def after(cls, seconds: float) -> OperationDeadline:
        return cls(expires_at=monotonic() + seconds, cancelled=Event())

    @classmethod
    def never(cls) -> OperationDeadline:
        return cls(expires_at=float("inf"), cancelled=Event())

    def cancel(self) -> None:
        self.cancelled.set()

    def check(self) -> None:
        if self.cancelled.is_set() or monotonic() >= self.expires_at:
            raise OperationCancelledError("operation deadline exceeded")

    def remaining(self, maximum: float) -> float:
        self.check()
        return max(0.001, min(maximum, self.expires_at - monotonic()))
