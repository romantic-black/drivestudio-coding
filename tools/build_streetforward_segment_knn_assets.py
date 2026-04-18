from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf

# Allow direct script execution without requiring manual PYTHONPATH export.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.streetforward_assets import StreetForwardAssetStore
from tools.streetforward_export_require_full_config import (
    require_full_training_config_for_asset_export,
)

try:
    from sklearn.neighbors import NearestNeighbors
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "build_streetforward_segment_knn_assets.py requires scikit-learn "
        "(import sklearn.neighbors.NearestNeighbors failed)"
    ) from exc


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _resolve_knn_modes(cfg: Any) -> Tuple[List[int], List[int], Dict[str, int]]:
    model_cfg = _cfg_get(cfg, "model")
    branches = _cfg_get(model_cfg, "branches")
    if branches is None:
        raise ValueError("model.branches is required")

    branch_k: Dict[str, int] = {}
    for branch_name in ("bg", "distant", "rigid"):
        branch = _cfg_get(branches, branch_name)
        if branch is None:
            continue
        init_cfg = _cfg_get(branch, "init")
        scale_init = _cfg_get(init_cfg, "scale_init")
        if scale_init is None:
            continue
        mode = str(_cfg_get(scale_init, "mode", "isotropic"))
        if mode != "knn":
            continue
        k = int(_cfg_get(scale_init, "knn_k", 0))
        if k <= 0:
            raise ValueError(
                f"model.branches.{branch_name}.init.scale_init.knn_k must be > 0 when mode=knn, got {k}"
            )
        branch_k[branch_name] = int(k)

    bg_distant_ks = sorted(
        {int(branch_k[name]) for name in ("bg", "distant") if name in branch_k}
    )
    rigid_ks = sorted({int(branch_k["rigid"])}) if "rigid" in branch_k else []
    return bg_distant_ks, rigid_ks, branch_k


def _avg_knn_distance(points_xyz: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.ones((1,), dtype=np.float32)
    k_eff = min(max(int(k), 1), n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
    nn.fit(pts)
    dists, _ = nn.kneighbors(pts)
    avg = dists[:, 1:].mean(axis=1)
    return np.asarray(avg, dtype=np.float32)


def _build_knn_payload_from_pointcloud(
    pointcloud: Dict[str, Any],
    *,
    bg_distant_ks: Sequence[int],
    rigid_ks: Sequence[int],
) -> Dict[str, Any]:
    background = np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)), dtype=np.float32)
    if background.ndim != 2 or background.shape[1] < 3:
        raise ValueError(f"pointcloud.background must have shape [N,>=3], got {tuple(background.shape)}")
    bg_xyz = background[:, :3].astype(np.float32, copy=False)
    background_avg_dist_by_k: Dict[int, np.ndarray] = {}
    for k in sorted(int(x) for x in bg_distant_ks):
        background_avg_dist_by_k[int(k)] = _avg_knn_distance(bg_xyz, int(k))

    dynamic_raw = pointcloud.get("dynamic", {})
    if not isinstance(dynamic_raw, dict):
        raise ValueError("pointcloud.dynamic must be a dict[intid -> np.ndarray]")
    dynamic_instance_ids = sorted(int(x) for x in dynamic_raw.keys())

    dynamic_avg_dist_by_k: Dict[int, Dict[int, np.ndarray]] = {}
    for k in sorted(int(x) for x in rigid_ks):
        per_instance: Dict[int, np.ndarray] = {}
        for intid in dynamic_instance_ids:
            arr = np.asarray(dynamic_raw[int(intid)], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(
                    f"pointcloud.dynamic[{intid}] must have shape [N,>=3], got {tuple(arr.shape)}"
                )
            per_instance[int(intid)] = _avg_knn_distance(arr[:, :3], int(k))
        dynamic_avg_dist_by_k[int(k)] = per_instance

    return {
        "background_avg_dist_by_k": background_avg_dist_by_k,
        "dynamic_avg_dist_by_k": dynamic_avg_dist_by_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build StreetForward segment kNN init assets")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    require_full_training_config_for_asset_export(cfg)

    data_cfg = _cfg_get(cfg, "data")
    assets_cfg = _cfg_get(data_cfg, "assets")
    if assets_cfg is None or _cfg_get(assets_cfg, "root") is None:
        raise ValueError("data.assets.root is required")

    dataset_name = str(_cfg_get(data_cfg, "dataset"))
    if not dataset_name:
        raise ValueError("data.dataset is required")

    bg_distant_ks, rigid_ks, branch_k = _resolve_knn_modes(cfg)
    if len(bg_distant_ks) == 0 and len(rigid_ks) == 0:
        print("[segment-knn] skipped=true reason=no_branch_uses_knn")
        return

    print(
        "[segment-knn] config "
        f"bg_distant_ks={bg_distant_ks} rigid_ks={rigid_ks} branch_k={branch_k} overwrite={bool(args.overwrite)}"
    )

    store = StreetForwardAssetStore(str(_cfg_get(assets_cfg, "root")), missing_policy="error")
    scene_ids_cfg = [int(x) for x in list(_cfg_get(data_cfg, "train_scene_ids", []) or [])]
    scene_ids = scene_ids_cfg if scene_ids_cfg else store.list_registered_scene_ids(dataset_name)
    if len(scene_ids) == 0:
        raise ValueError(
            f"No scene ids to process for dataset={dataset_name}: "
            "data.train_scene_ids empty and segment_registry has no rows"
        )

    for scene_id in scene_ids:
        segment_ids = store.list_registered_segment_ids(dataset_name, int(scene_id))
        if len(segment_ids) == 0:
            print(f"[segment-knn] scene_id={scene_id} skipped=true reason=no_registered_segments")
            continue
        for seg_id in segment_ids:
            segment_handle = store.get_segment_asset_registry_first(dataset_name, int(scene_id), int(seg_id))
            if segment_handle.has_knn_init() and not bool(args.overwrite):
                print(
                    f"[segment-knn] scene_id={scene_id} segment_id={seg_id} "
                    "status=existing overwrite=false"
                )
                continue

            pointcloud = segment_handle.load_pointcloud()
            payload = _build_knn_payload_from_pointcloud(
                pointcloud,
                bg_distant_ks=bg_distant_ks,
                rigid_ks=rigid_ks,
            )
            written = store.export_segment_knn_init_asset(
                dataset=dataset_name,
                scene_id=int(scene_id),
                segment_id=int(seg_id),
                knn_payload=payload,
                overwrite=bool(args.overwrite),
            )
            bg_n = int(np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)).shape[0]))
            dyn_n = int(sum(int(np.asarray(v).shape[0]) for v in pointcloud.get("dynamic", {}).values()))
            print(
                f"[segment-knn] scene_id={scene_id} segment_id={seg_id} "
                f"status={'written' if written else 'existing'} "
                f"background_points={bg_n} dynamic_points={dyn_n}"
            )


if __name__ == "__main__":
    main()
