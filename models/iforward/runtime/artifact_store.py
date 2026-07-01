from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("images", "grids", "videos", "plots"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def relpath(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.root.resolve()))

    def save_json(self, name: str, obj: dict[str, Any]) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return self.relpath(path)

    def save_image(self, name: str, tensor_or_array: Any) -> str:
        path = self.root / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        array = _to_uint8_hwc(tensor_or_array)
        try:
            from PIL import Image

            Image.fromarray(array).save(path)
        except Exception:
            import imageio.v2 as imageio

            imageio.imwrite(path, array)
        return self.relpath(path)

    def save_grid(self, name: str, images: list[Any]) -> str:
        if not images:
            raise ValueError("ArtifactStore.save_grid requires at least one image")
        tensors = [_to_chw_float(image) for image in images]
        max_h = max(int(t.shape[-2]) for t in tensors)
        padded = []
        for tensor in tensors:
            if int(tensor.shape[-2]) != max_h:
                pad = max_h - int(tensor.shape[-2])
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad), value=1.0)
            padded.append(tensor)
        return self.save_image(str(Path("..") / "grids" / name), torch.cat(padded, dim=-1))

    def save_video(self, name: str, frames: list[Any]) -> str:
        path = self.root / "videos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        imageio.mimsave(path, [_to_uint8_hwc(frame) for frame in frames])
        return self.relpath(path)


def _to_chw_float(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        image = value.detach().float().cpu()
    else:
        image = torch.as_tensor(value).float().cpu()
    while int(image.ndim) > 3:
        image = image[0]
    if int(image.ndim) == 2:
        image = image.unsqueeze(0)
    if int(image.ndim) != 3:
        raise ValueError(f"expected 2D/3D image tensor, got shape={tuple(image.shape)}")
    if int(image.shape[0]) in {1, 3}:
        out = image
    elif int(image.shape[-1]) in {1, 3}:
        out = image.permute(2, 0, 1)
    else:
        raise ValueError(f"expected CHW/HWC image with 1 or 3 channels, got shape={tuple(image.shape)}")
    if int(out.shape[0]) == 1:
        out = out.repeat(3, 1, 1)
    return out.clamp(0.0, 1.0)


def _to_uint8_hwc(value: Any) -> Any:
    image = _to_chw_float(value)
    image = (image.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).round().byte()
    return image.numpy()


__all__ = ["ArtifactStore"]
