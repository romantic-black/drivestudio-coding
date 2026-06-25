from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class ProtocolDeficitSampler:
    weights: Dict[str, float]
    cursor: int = 0

    def __post_init__(self) -> None:
        clean = {str(k): float(v) for k, v in dict(self.weights or {}).items() if float(v) > 0.0}
        if not clean:
            clean = {"D1": 1.0, "D2": 1.0, "I123": 1.0}
        self.weights = clean
        expanded: List[str] = []
        for name, weight in sorted(clean.items()):
            count = max(1, int(round(float(weight) * 10.0)))
            expanded.extend([str(name)] * int(count))
        self._expanded = expanded

    def next(self, available: Optional[Iterable[str]] = None) -> str:
        allowed = None if available is None else {str(x) for x in available}
        if allowed is None:
            allowed = set(self.weights)
        if not allowed:
            raise ValueError("Stage2_2 protocol sampler has no available protocols")
        for _ in range(max(1, len(self._expanded))):
            name = str(self._expanded[int(self.cursor) % len(self._expanded)])
            self.cursor = int(self.cursor) + 1
            if name in allowed:
                return name
        return sorted(allowed)[0]

    def state_dict(self) -> Dict[str, object]:
        return {"weights": dict(self.weights), "cursor": int(self.cursor)}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.cursor = int(state.get("cursor", 0))


__all__ = ["ProtocolDeficitSampler"]
