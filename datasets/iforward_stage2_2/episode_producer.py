from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Optional


@dataclass
class EpisodeProducer:
    """Small bounded-queue facade for Stage2_2 episode plans.

    The first implementation is intentionally synchronous unless callers pass a
    background worker around it. The scheduler state never serializes queued
    plans, which keeps resume deterministic and mmap-only.
    """

    maxsize: int = 2
    queue: Queue = field(init=False)
    exception: Optional[BaseException] = None

    def __post_init__(self) -> None:
        self.queue = Queue(maxsize=max(1, int(self.maxsize)))

    def put(self, item: Any) -> None:
        self.queue.put(item)

    def get_or_build(self, builder: Callable[[], Any]) -> Any:
        if self.exception is not None:
            raise RuntimeError("Stage2_2 episode producer failed") from self.exception
        if not self.queue.empty():
            return self.queue.get()
        try:
            return builder()
        except BaseException as exc:
            self.exception = exc
            raise


__all__ = ["EpisodeProducer"]
