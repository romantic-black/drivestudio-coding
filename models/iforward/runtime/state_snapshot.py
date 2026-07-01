from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeSnapshot:
    name: str
    carried_state: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def clone_runtime_state(state: Any) -> Any:
    detach = getattr(state, "detach_for_next_rollout", None)
    return detach() if callable(detach) else state


__all__ = ["RuntimeSnapshot", "clone_runtime_state"]
