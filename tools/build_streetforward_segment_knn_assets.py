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


def _expand_topk_ks(requested_ks: Sequence[int]) -> List[int]:
    if len(requested_ks) == 0:
        return []
    k_max = max(int(x) for x in requested_ks)
    if k_max <= 0:
        raise ValueError(f"knn_k must be > 0, got {k_max}")
    return list(range(1, int(k_max) + 1))


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


def _avg_knn_distance_prefix(points_xyz: np.ndarray, max_k: int) -> Dict[int, np.ndarray]:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    k_max = int(max_k)
    if k_max <= 0:
        raise ValueError(f"max_k must be > 0, got {k_max}")
    if n == 0:
        return {k: np.zeros((0,), dtype=np.float32) for k in range(1, k_max + 1)}
    if n == 1:
        return {k: np.ones((1,), dtype=np.float32) for k in range(1, k_max + 1)}

    k_eff_max = min(k_max, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff_max + 1, algorithm="auto", metric="euclidean")
    nn.fit(pts)
    dists, _ = nn.kneighbors(pts)
    d = np.asarray(dists[:, 1:], dtype=np.float32)  # [N, k_eff_max]
    csum = np.cumsum(d, axis=1)  # [N, k_eff_max]

    out: Dict[int, np.ndarray] = {}
    for k in range(1, k_max + 1):
        k_eff = min(int(k), n - 1)
        out[int(k)] = (csum[:, k_eff - 1] / float(k_eff)).astype(np.float32, copy=False)
    return out


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
    bg_ks_sorted = sorted(int(x) for x in bg_distant_ks)
    if len(bg_ks_sorted) > 0:
        bg_prefix = _avg_knn_distance_prefix(bg_xyz, max_k=max(bg_ks_sorted))
        for k in bg_ks_sorted:
            background_avg_dist_by_k[int(k)] = bg_prefix[int(k)]

    dynamic_raw = pointcloud.get("dynamic", {})
    if not isinstance(dynamic_raw, dict):
        raise ValueError("pointcloud.dynamic must be a dict[intid -> np.ndarray]")
    dynamic_instance_ids = sorted(int(x) for x in dynamic_raw.keys())

    dynamic_avg_dist_by_k: Dict[int, Dict[int, np.ndarray]] = {}
    rigid_ks_sorted = sorted(int(x) for x in rigid_ks)
    for k in rigid_ks_sorted:
        dynamic_avg_dist_by_k[int(k)] = {}
    if len(rigid_ks_sorted) > 0:
        max_k = max(rigid_ks_sorted)
        for intid in dynamic_instance_ids:
            arr = np.asarray(dynamic_raw[int(intid)], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(
                    f"pointcloud.dynamic[{intid}] must have shape [N,>=3], got {tuple(arr.shape)}"
                )
            pref = _avg_knn_distance_prefix(arr[:, :3], max_k=max_k)
            for k in rigid_ks_sorted:
                dynamic_avg_dist_by_k[int(k)][int(intid)] = pref[int(k)]

    return {
        "background_avg_dist_by_k": background_avg_dist_by_k,
        "dynamic_avg_dist_by_k": dynamic_avg_dist_by_k,
    }


def _existing_knn_covers_required_ks(
    existing_knn: Any,
    *,
    bg_distant_ks: Sequence[int],
    rigid_ks: Sequence[int],
) -> bool:
    if not isinstance(existing_knn, dict):
        return False
    bg_map = existing_knn.get("background_avg_dist_by_k", {})
    dyn_map = existing_knn.get("dynamic_avg_dist_by_k", {})
    if not isinstance(bg_map, dict) or not isinstance(dyn_map, dict):
        return False
    bg_have = {int(k) for k in bg_map.keys()}
    dyn_have = {int(k) for k in dyn_map.keys()}
    return all(int(k) in bg_have for k in bg_distant_ks) and all(int(k) in dyn_have for k in rigid_ks)


def _resolve_target_scene_ids(data_cfg: Any, store: StreetForwardAssetStore, dataset_name: str) -> List[int]:
    train_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "train_scene_ids", []) or [])]
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    merged = sorted(set(train_scene_ids + eval_scene_ids))
    if len(merged) > 0:
        return merged
    return store.list_registered_scene_ids(dataset_name)


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

    bg_distant_ks_req, rigid_ks_req, branch_k = _resolve_knn_modes(cfg)
    if len(bg_distant_ks_req) == 0 and len(rigid_ks_req) == 0:
        print("[segment-knn] skipped=true reason=no_branch_uses_knn")
        return
    bg_distant_ks = _expand_topk_ks(bg_distant_ks_req)
    rigid_ks = _expand_topk_ks(rigid_ks_req)

    print(
        "[segment-knn] config "
        f"branch_k={branch_k} "
        f"requested_bg_distant_ks={bg_distant_ks_req} requested_rigid_ks={rigid_ks_req} "
        f"export_bg_distant_ks={bg_distant_ks} export_rigid_ks={rigid_ks} "
        f"overwrite={bool(args.overwrite)}"
    )

    store = StreetForwardAssetStore(str(_cfg_get(assets_cfg, "root")), missing_policy="error")
    scene_ids = _resolve_target_scene_ids(data_cfg, store, dataset_name)
    if len(scene_ids) == 0:
        raise ValueError(
            f"No scene ids to process for dataset={dataset_name}: "
            "data.train_scene_ids/data.eval_scene_ids are empty and segment_registry has no rows"
        )
    print(f"[segment-knn] target_scenes={len(scene_ids)} ids={scene_ids}")

    for scene_id in scene_ids:
        segment_ids = store.list_registered_segment_ids(dataset_name, int(scene_id))
        if len(segment_ids) == 0:
            print(f"[segment-knn] scene_id={scene_id} skipped=true reason=no_registered_segments")
            continue
        for seg_id in segment_ids:
            segment_handle = store.get_segment_asset_registry_first(dataset_name, int(scene_id), int(seg_id))
            has_knn = bool(segment_handle.has_knn_init())
            force_overwrite = bool(args.overwrite)
            refresh_missing_k = False
            if has_knn and not bool(args.overwrite):
                existing_knn = segment_handle.load_knn_init()
                if _existing_knn_covers_required_ks(
                    existing_knn,
                    bg_distant_ks=bg_distant_ks,
                    rigid_ks=rigid_ks,
                ):
                    print(
                        f"[segment-knn] scene_id={scene_id} segment_id={seg_id} "
                        "status=existing overwrite=false"
                    )
                    continue
                refresh_missing_k = True
                force_overwrite = True
                print(
                    f"[segment-knn] scene_id={scene_id} segment_id={seg_id} "
                    "status=refresh reason=missing_required_k overwrite=false"
                )

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
                overwrite=bool(force_overwrite),
            )
            if refresh_missing_k and not bool(written):
                raise RuntimeError(
                    f"Internal error: refresh requested but KNN write was skipped "
                    f"(scene_id={scene_id} segment_id={seg_id})."
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
