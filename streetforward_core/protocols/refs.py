from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Tuple


@dataclass(frozen=True, order=True)
class ImageRef:
    frame_idx: int
    cam_idx: int

    @staticmethod
    def from_raw(x: Any) -> "ImageRef":
        if isinstance(x, ImageRef):
            return x
        if isinstance(x, dict) and "frame_idx" in x and "cam_idx" in x:
            return ImageRef(frame_idx=int(x["frame_idx"]), cam_idx=int(x["cam_idx"]))
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"ImageRef requires length 2, got {x!r}")
        return ImageRef(frame_idx=int(x[0]), cam_idx=int(x[1]))

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.frame_idx), int(self.cam_idx))

    def __iter__(self) -> Iterator[int]:
        yield int(self.frame_idx)
        yield int(self.cam_idx)
