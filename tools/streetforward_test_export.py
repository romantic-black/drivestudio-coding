from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch


def save_3dgs_state(path: str, state: Dict[str, Any]) -> None:
    torch.save(state, path)


def _extract_branch_for_ply(state: Dict[str, Any], branch_name: str) -> Optional[Dict[str, torch.Tensor]]:
    branches = state.get("branches")
    if not isinstance(branches, dict):
        raise ValueError("state.branches is required")
    b = branches.get(branch_name)
    if b is None:
        return None
    required = ("means", "sh_dc", "opacity_logit")
    for k in required:
        if k not in b:
            raise ValueError(f"state.branches.{branch_name}.{k} is required for ply export")
    return b


def _branch_rgb_from_sh_dc(branch: Dict[str, torch.Tensor]) -> torch.Tensor:
    sh_dc = branch["sh_dc"].detach().cpu().float()
    rgb = torch.clamp(sh_dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return rgb


def _collect_vertices(state: Dict[str, Any]) -> List[str]:
    branches: List[Dict[str, torch.Tensor]] = []
    for name in ("bg", "distant", "sky"):
        b = _extract_branch_for_ply(state, name)
        if b is not None:
            branches.append(b)

    rigid_world = _extract_branch_for_ply(state, "rigid_world")
    if rigid_world is None and _extract_branch_for_ply(state, "rigid_local") is not None:
        raise ValueError(
            "save_3dgs_ply requires rigid_world branch. "
            "Do not export rigid_local directly into merged scene PLY."
        )
    if rigid_world is not None:
        branches.append(rigid_world)

    vertices: List[str] = []
    for b in branches:
        means = b["means"].detach().cpu().float()
        opacity = torch.sigmoid(b["opacity_logit"].detach().cpu().float()).squeeze(-1)
        rgb = _branch_rgb_from_sh_dc(b)
        n = int(means.shape[0])
        for i in range(n):
            x, y, z = means[i].tolist()
            r, g, bb = rgb[i].tolist()
            a = float(opacity[i].item())
            r8 = int(max(0, min(255, round(r * 255.0))))
            g8 = int(max(0, min(255, round(g * 255.0))))
            b8 = int(max(0, min(255, round(bb * 255.0))))
            vertices.append(f"{x:.6f} {y:.6f} {z:.6f} {r8} {g8} {b8} {a:.6f}")
    return vertices


def save_3dgs_ply(path: str, state: Dict[str, Any]) -> None:
    vertices = _collect_vertices(state)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "end_header",
    ]
    lines.extend(vertices)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_test_summary(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
