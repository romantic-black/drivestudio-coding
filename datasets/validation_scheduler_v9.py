from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

from datasets.train_scheduler_v9 import ImageRef, StepPlanV9, ViewSetRolloutBatchV9


ValidationPhaseV9 = Literal[
    "phase_A_block_local_unroll",
    "phase_B_viewset_rollout",
]


class SegmentIndexLike(Protocol):
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    frame_to_keyframe: Dict[int, int]
    num_cams: int


class ValidationDatasetLike(Protocol):
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...


@dataclass(frozen=True)
class ValidationBlockSpecV9:
    phase: ValidationPhaseV9

    scene_id: int
    segment_id: int
    segment_choice_rank: int

    episode_start_keyframe_pos: int
    keyframe_window: List[int]
    frame_chain: List[int]

    block_idx: int
    source_keyframe_idx: int
    source_frame_idx: int

    evidence_refs: List[ImageRef]
    block_loss_refs: List[ImageRef]
    nearby_loss_refs: List[ImageRef]

    prefix_loss_refs_by_step: List[List[ImageRef]] = field(default_factory=list)
    query_label_refs: List[ImageRef] = field(default_factory=list)
    aux_loss_refs: List[ImageRef] = field(default_factory=list)

    num_cams: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationPlanV9:
    scheduler_version: str
    phase: ValidationPhaseV9

    seed: int
    eval_scene_ids: List[int]
    blocks_per_episode: int

    k_values: List[int]
    max_K: int

    block_specs: List[ValidationBlockSpecV9]

    request_meta: Dict[str, Any] = field(default_factory=dict)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        try:
            value = node.get(key, default)
            return default if value is None else value
        except TypeError:
            pass
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def _cfg_path(node: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = node
    for key in path:
        cur = _cfg_get(cur, key, None)
        if cur is None:
            return default
    return cur


def _flat_or_path(cfg: Any, flat_key: str, path: Sequence[str], default: Any = None) -> Any:
    value = _cfg_get(cfg, flat_key, None)
    if value is not None:
        return value
    return _cfg_path(cfg, path, default)


def _all_cams(frame_idx: int, num_cams: int) -> List[ImageRef]:
    return [(int(frame_idx), int(cam_idx)) for cam_idx in range(int(num_cams))]


def _dedupe_refs_keep_order(refs: Sequence[ImageRef]) -> List[ImageRef]:
    seen: set[ImageRef] = set()
    out: List[ImageRef] = []
    for ref in refs:
        r = (int(ref[0]), int(ref[1]))
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _dedupe_refs_roles_keep_order(
    refs: Sequence[ImageRef],
    roles: Sequence[str],
) -> tuple[List[ImageRef], List[str]]:
    if len(refs) != len(roles):
        raise ValueError(f"refs/roles length mismatch: {len(refs)} vs {len(roles)}")
    seen: set[ImageRef] = set()
    out_refs: List[ImageRef] = []
    out_roles: List[str] = []
    for ref, role in zip(refs, roles):
        r = (int(ref[0]), int(ref[1]))
        if r in seen:
            continue
        seen.add(r)
        out_refs.append(r)
        out_roles.append(str(role))
    return out_refs, out_roles


def _flatten(ref_groups: Sequence[Sequence[ImageRef]]) -> List[ImageRef]:
    out: List[ImageRef] = []
    for group in ref_groups:
        for ref in group:
            out.append((int(ref[0]), int(ref[1])))
    return out


def _middle_frame(frames: Sequence[int]) -> int:
    vals = [int(x) for x in list(frames)]
    if not vals:
        raise ValueError("keyframe_to_frames entry must not be empty")
    vals.sort()
    return int(vals[len(vals) // 2])


def _episode_starts(num_keyframes: int, blocks_per_episode: int) -> List[int]:
    window = int(blocks_per_episode)
    if int(num_keyframes) < 1:
        return []
    if int(num_keyframes) < window:
        return [0]
    starts = list(range(0, int(num_keyframes) - window + 1, int(blocks_per_episode)))
    tail = int(num_keyframes) - window
    if starts[-1] != tail:
        starts.append(tail)
    return [int(x) for x in starts]


def choose_segment_for_scene(
    dataset: ValidationDatasetLike,
    scene_id: int,
    *,
    seed: int,
    policy: str,
) -> Optional[int]:
    seg_ids = sorted(int(x) for x in list(dataset.list_segment_ids(int(scene_id))))
    if not seg_ids:
        return None
    policy = str(policy)
    if policy == "first":
        return int(seg_ids[0])
    if policy == "middle":
        return int(seg_ids[len(seg_ids) // 2])
    if policy == "random_seeded":
        rng = random.Random(int(seed) + int(scene_id) * 10007)
        return int(rng.choice(seg_ids))
    raise ValueError(f"unsupported validation_v9 segment_policy={policy!r}")


def choose_episode_start(
    *,
    num_keyframes: int,
    blocks_per_episode: int,
    seed: int,
    scene_id: int,
    segment_id: int,
    policy: str,
) -> Optional[int]:
    starts = _episode_starts(int(num_keyframes), int(blocks_per_episode))
    if not starts:
        return None
    policy = str(policy)
    if policy == "first":
        return int(starts[0])
    if policy == "middle":
        return int(starts[len(starts) // 2])
    if policy == "random_seeded":
        rng = random.Random(int(seed) + int(scene_id) * 10007 + int(segment_id) * 1009)
        return int(rng.choice(starts))
    raise ValueError(f"unsupported validation_v9 episode_policy={policy!r}")


def choose_blocks(
    *,
    blocks_per_episode: int,
    n: int,
    seed: int,
    scene_id: int,
    segment_id: int,
    policy: str,
) -> List[int]:
    blocks = list(range(int(blocks_per_episode)))
    count = min(int(n), len(blocks))
    if count < 1:
        raise ValueError("validation_v9 selection.blocks_per_segment must be >= 1")
    policy = str(policy)
    if policy == "first_n":
        return [int(x) for x in blocks[:count]]
    if policy == "middle_n":
        start = max((len(blocks) - count) // 2, 0)
        return [int(x) for x in blocks[start : start + count]]
    if policy == "random_without_replacement":
        rng = random.Random(int(seed) + int(scene_id) * 10007 + int(segment_id) * 1009)
        rng.shuffle(blocks)
        return [int(x) for x in blocks[:count]]
    raise ValueError(f"unsupported validation_v9 block_policy={policy!r}")


def choose_source_frame(
    frames: Sequence[int],
    *,
    seed: int,
    scene_id: int,
    segment_id: int,
    keyframe_idx: int,
    policy: str,
) -> int:
    vals = sorted(int(x) for x in list(frames))
    if not vals:
        raise ValueError("keyframe_to_frames entry must not be empty")
    policy = str(policy)
    if policy == "middle_in_keyframe":
        return int(vals[len(vals) // 2])
    if policy == "first_in_keyframe":
        return int(vals[0])
    if policy == "random_seeded":
        rng = random.Random(
            int(seed)
            + int(scene_id) * 10007
            + int(segment_id) * 1009
            + int(keyframe_idx) * 101
        )
        return int(rng.choice(vals))
    raise ValueError(f"unsupported validation_v9 source_frame_policy={policy!r}")


def sample_nearby_frames_for_validation(
    *,
    sidx: SegmentIndexLike,
    source_keyframe_idx: int,
    source_frame: int,
    existing_loss_frames: Sequence[int],
    num_frames: int,
    policy: str,
    adjacent_radius: int,
    random_fill: bool,
    seed: int,
    scene_id: int,
    segment_id: int,
) -> List[int]:
    if int(num_frames) <= 0:
        return []
    if str(policy) != "adjacent_then_random_same_keyframe":
        raise ValueError(f"unsupported validation_v9 nearby policy={policy!r}")
    frames = sorted(int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)]))
    if not frames:
        return []
    excluded = {int(source_frame)}
    excluded.update(int(x) for x in existing_loss_frames)
    out: List[int] = []
    if int(source_frame) in frames:
        pos = frames.index(int(source_frame))
        for radius in range(1, int(adjacent_radius) + 1):
            for cand_pos in (pos - radius, pos + radius):
                if cand_pos < 0 or cand_pos >= len(frames):
                    continue
                cand = int(frames[cand_pos])
                if cand in excluded or cand in out:
                    continue
                out.append(cand)
                if len(out) >= int(num_frames):
                    return [int(x) for x in out]
    if bool(random_fill) and len(out) < int(num_frames):
        rng = random.Random(
            int(seed)
            + int(scene_id) * 10007
            + int(segment_id) * 1009
            + int(source_keyframe_idx) * 101
            + int(source_frame) * 17
        )
        candidates = [int(f) for f in frames if int(f) not in excluded and int(f) not in out]
        rng.shuffle(candidates)
        out.extend(candidates[: max(int(num_frames) - len(out), 0)])
    return [int(x) for x in out[: int(num_frames)]]


def build_request_meta_v9_for_validation(
    plan: ViewSetRolloutBatchV9,
    *,
    k_values: Sequence[int],
    max_K: int,
    block_loss_mask: str = "non_sky_non_egocar",
    nearby_loss_mask: str = "non_sky_non_egocar",
    phase_b_vsm_scope: str = "bg_static",
) -> Dict[str, Any]:
    evidence_refs = _dedupe_refs_keep_order(_flatten(plan.evidence_refs_by_step))
    block_refs_raw = _flatten(plan.block_loss_refs_by_step)
    nearby_refs_raw = _flatten(plan.nearby_loss_refs_by_step)
    prefix_refs_raw = _flatten(plan.prefix_loss_refs_by_step)
    render_refs_raw = block_refs_raw + nearby_refs_raw + prefix_refs_raw
    render_roles_raw = (
        ["block_loss" for _ in block_refs_raw]
        + ["nearby_loss" for _ in nearby_refs_raw]
        + ["prefix_loss" for _ in prefix_refs_raw]
    )
    render_refs, render_roles = _dedupe_refs_roles_keep_order(render_refs_raw, render_roles_raw)
    query_refs = _dedupe_refs_keep_order([tuple(x) for x in plan.query_label_refs])
    aux_refs = _dedupe_refs_keep_order([tuple(x) for x in plan.aux_loss_refs])
    non_evidence_refs = _dedupe_refs_keep_order(render_refs + query_refs + aux_refs)
    nearby_frames = sorted({int(ref[0]) for ref in nearby_refs_raw})
    query_frames = sorted({int(ref[0]) for ref in query_refs})
    leakage_check = {
        "nearby_evidence_overlap": int(len(set(nearby_refs_raw) & set(evidence_refs))),
        "query_evidence_overlap": int(len(set(query_refs) & set(evidence_refs))),
        "aux_evidence_overlap": int(len(set(aux_refs) & set(evidence_refs))),
        "same_scene_segment_required": True,
        "num_evidence_refs": int(len(evidence_refs)),
        "num_render_loss_refs": int(len(render_refs)),
        "num_query_label_refs": int(len(query_refs)),
        "target_role_count_match": bool(len(render_refs) == len(render_roles)),
    }
    mask_policy = {
        "phase_a_block_loss_mask": str(block_loss_mask),
        "phase_a_nearby_loss_mask": str(nearby_loss_mask),
        "phase_b_vsm_scope": str(phase_b_vsm_scope),
        "phase_b_evidence_mask": "valid_non_sky_non_egocar_non_dynamic",
        "phase_b_prefix_loss_mask": "valid_non_sky_non_egocar_non_dynamic",
        "phase_b_query_label_mask": "valid_non_sky_non_egocar_non_dynamic",
    }
    return {
        "scheduler_version": "v9",
        "scheduler_phase": str(plan.phase),
        "scene_id": int(plan.scene_id),
        "segment_id": int(plan.segment_id),
        "episode_id": int(plan.episode_id),
        "episode_idx_global": int(plan.episode_id),
        "episode_start_keyframe_pos": int(plan.episode_start_keyframe_pos),
        "inner_K": int(plan.inner_K),
        "evidence_refs_by_step": [[tuple(x) for x in refs] for refs in plan.evidence_refs_by_step],
        "block_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.block_loss_refs_by_step],
        "nearby_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.nearby_loss_refs_by_step],
        "prefix_loss_refs_by_step": [[tuple(x) for x in refs] for refs in plan.prefix_loss_refs_by_step],
        "query_label_refs": [tuple(x) for x in query_refs],
        "aux_loss_refs": [tuple(x) for x in aux_refs],
        "flat_evidence_refs": [tuple(x) for x in evidence_refs],
        "flat_render_loss_refs": [tuple(x) for x in render_refs],
        "flat_non_evidence_refs": [tuple(x) for x in non_evidence_refs],
        "flat_loss_refs": [tuple(x) for x in non_evidence_refs],
        "source_image_refs": [tuple(x) for x in evidence_refs],
        "source_image_ref": tuple(evidence_refs[0]) if evidence_refs else None,
        "target_image_refs": [tuple(x) for x in render_refs],
        "target_image_roles": [str(x) for x in render_roles],
        "nearby_loss_frame_indices": [int(x) for x in nearby_frames],
        "nearby_frame_indices": [int(x) for x in nearby_frames],
        "query_label_frame_indices": [int(x) for x in query_frames],
        "role_policy": {
            "evidence": "update_only",
            "block_loss": "loss_only",
            "nearby_loss": "loss_only",
            "prefix_loss": "loss_only",
            "query_label": "label_only",
            "aux_loss": "loss_only",
        },
        "mask_policy": mask_policy,
        "vsm_reset_policy": {"reset_vsm_on_episode_end": True, "episode_id": int(plan.episode_id)},
        "role_groups": [
            {
                "role": "evidence",
                "refs": [tuple(x) for x in evidence_refs],
                "image_roles": ["evidence" for _ in evidence_refs],
                "allow_update_evidence": True,
                "allow_render_loss": False,
                "allow_query_label": False,
                "mask_policy": "valid_non_sky_non_egocar_non_dynamic",
            },
            {
                "role": "render_loss",
                "refs": [tuple(x) for x in render_refs],
                "image_roles": [str(x) for x in render_roles],
                "allow_update_evidence": False,
                "allow_render_loss": True,
                "allow_query_label": False,
                "mask_policy": mask_policy,
            },
            {
                "role": "query_label",
                "refs": [tuple(x) for x in query_refs],
                "image_roles": ["query_label" for _ in query_refs],
                "allow_update_evidence": False,
                "allow_render_loss": False,
                "allow_query_label": True,
                "mask_policy": "valid_non_sky_non_egocar_non_dynamic",
            },
        ],
        "leakage_check": leakage_check,
        "assembly_mode": "image_ref_v9",
        "validation_version": "v9",
        "validation_mode": "phase_a_k_sweep",
        "validation_phase": "phase_A_block_local_unroll",
        "validation_k_values": [int(x) for x in k_values],
        "validation_max_K": int(max_K),
    }


def validate_phase_a_rollout_plan_v9(
    plan: ViewSetRolloutBatchV9,
    *,
    dataset: Optional[ValidationDatasetLike] = None,
    fail_fast: bool = True,
) -> None:
    if str(plan.scheduler_version) != "v9":
        raise ValueError("expected scheduler_version=v9")
    if str(plan.phase) != "phase_A_block_local_unroll":
        raise ValueError("validation_v9 P0 only supports phase_A_block_local_unroll")
    if int(plan.inner_K) < 1:
        raise ValueError("validation_v9 inner_K must be >= 1")
    if len(plan.steps) != int(plan.inner_K):
        raise ValueError("len(steps) must equal inner_K")
    for attr in (
        "evidence_refs_by_step",
        "block_loss_refs_by_step",
        "nearby_loss_refs_by_step",
        "prefix_loss_refs_by_step",
    ):
        if len(getattr(plan, attr)) != int(plan.inner_K):
            raise ValueError(f"len({attr}) must equal inner_K")
    evidence = set(_flatten(plan.evidence_refs_by_step))
    nearby = set(_flatten(plan.nearby_loss_refs_by_step))
    prefix = set(_flatten(plan.prefix_loss_refs_by_step))
    query = set(tuple(x) for x in plan.query_label_refs)
    aux = set(tuple(x) for x in plan.aux_loss_refs)
    if not evidence:
        raise ValueError("validation_v9 requires non-empty evidence_refs")
    if nearby & evidence:
        raise ValueError("nearby_loss_refs must not overlap evidence_refs")
    if prefix:
        raise ValueError("Phase A validation must not emit prefix refs")
    if query:
        raise ValueError("Phase A validation must not emit query label refs")
    if aux:
        raise ValueError("Phase A validation must not emit aux refs")
    if dataset is not None:
        sidx = dataset.get_segment_index(int(plan.scene_id), int(plan.segment_id))
        frame_to_keyframe = getattr(sidx, "frame_to_keyframe", {}) or {}
        for step in plan.steps:
            for ref in step.nearby_loss_refs:
                actual = int(frame_to_keyframe.get(int(ref[0]), -1))
                if actual != int(step.source_keyframe_idx):
                    msg = "Phase A validation nearby frame is not in source keyframe"
                    if bool(fail_fast):
                        raise ValueError(msg)


def make_phase_a_eval_rollout_batch(
    spec: ValidationBlockSpecV9,
    *,
    max_K: int,
    k_values: Optional[Sequence[int]] = None,
    block_loss_mask: str = "non_sky_non_egocar",
    nearby_loss_mask: str = "non_sky_non_egocar",
) -> ViewSetRolloutBatchV9:
    if str(spec.phase) != "phase_A_block_local_unroll":
        raise ValueError("make_phase_a_eval_rollout_batch requires Phase A spec")
    if spec.prefix_loss_refs_by_step:
        raise ValueError("Phase A validation spec must not contain prefix refs")
    if spec.query_label_refs:
        raise ValueError("Phase A validation spec must not contain query refs")
    if spec.aux_loss_refs:
        raise ValueError("Phase A validation spec must not contain aux refs")
    if int(max_K) < 1:
        raise ValueError("validation_v9 max_K must be >= 1")
    steps: List[StepPlanV9] = []
    nearby_frames = sorted({int(ref[0]) for ref in spec.nearby_loss_refs})
    for k in range(int(max_K)):
        steps.append(
            StepPlanV9(
                step_idx=int(k),
                source_keyframe_idx=int(spec.source_keyframe_idx),
                source_frame_idx=int(spec.source_frame_idx),
                block_idx=int(spec.block_idx),
                evidence_refs=[tuple(x) for x in spec.evidence_refs],
                block_loss_refs=[tuple(x) for x in spec.block_loss_refs],
                nearby_loss_refs=[tuple(x) for x in spec.nearby_loss_refs],
                prefix_loss_refs=[],
                query_label_refs=[],
                aux_loss_refs=[],
                evidence_frame_indices=[int(spec.source_frame_idx)],
                loss_frame_indices=[int(spec.source_frame_idx)],
                nearby_frame_indices=[int(x) for x in nearby_frames],
                query_frame_indices=[],
            )
        )
    plan = ViewSetRolloutBatchV9(
        scheduler_version="v9",
        phase="phase_A_block_local_unroll",
        scene_id=int(spec.scene_id),
        segment_id=int(spec.segment_id),
        episode_id=-1,
        episode_start_keyframe_pos=int(spec.episode_start_keyframe_pos),
        keyframe_window=[int(x) for x in spec.keyframe_window],
        frame_chain=[int(x) for x in spec.frame_chain],
        num_cams=int(spec.num_cams),
        inner_K=int(max_K),
        steps=steps,
        evidence_refs_by_step=[[tuple(x) for x in s.evidence_refs] for s in steps],
        block_loss_refs_by_step=[[tuple(x) for x in s.block_loss_refs] for s in steps],
        nearby_loss_refs_by_step=[[tuple(x) for x in s.nearby_loss_refs] for s in steps],
        prefix_loss_refs_by_step=[[] for _ in steps],
        query_label_refs=[],
        aux_loss_refs=[],
        request_meta={
            "scheduler_version": "v9",
            "validation_version": "v9",
            "validation_phase": "phase_A_block_local_unroll",
            "validation_max_K": int(max_K),
            "validation_block_idx": int(spec.block_idx),
        },
    )
    meta = build_request_meta_v9_for_validation(
        plan,
        k_values=list(k_values or [0, int(max_K)]),
        max_K=int(max_K),
        block_loss_mask=str(block_loss_mask),
        nearby_loss_mask=str(nearby_loss_mask),
    )
    meta.update(dict(plan.request_meta or {}))
    meta["validation_k_values"] = [int(x) for x in list(k_values or [0, int(max_K)])]
    return dataclasses.replace(plan, request_meta=meta, leakage_check=dict(meta.get("leakage_check") or {}))


def build_validation_plan_v9(
    *,
    dataset: ValidationDatasetLike,
    eval_scene_ids: Sequence[int],
    cfg: Any,
    blocks_per_episode: Optional[int] = None,
) -> ValidationPlanV9:
    root_cfg = cfg
    nested_v9 = _cfg_get(cfg, "validation_v9", None)
    if nested_v9 is not None:
        cfg = nested_v9
    eval_ids = [int(x) for x in list(eval_scene_ids)]
    if not eval_ids:
        raise ValueError("validation_v9 requires non-empty eval_scene_ids.")
    phase = str(_flat_or_path(cfg, "phase", ["phase"], "phase_A_block_local_unroll"))
    if phase != "phase_A_block_local_unroll":
        raise ValueError("P0 validation_v9 only supports phase_A_block_local_unroll.")
    fail_fast = bool(_flat_or_path(cfg, "fail_fast", ["fail_fast"], True))
    seed = int(_flat_or_path(cfg, "selection_seed", ["selection", "seed"], 20260524))
    blocks_per_episode_val = blocks_per_episode
    if blocks_per_episode_val is None:
        blocks_per_episode_val = _flat_or_path(root_cfg, "blocks_per_episode", ["scheduler_v9", "episode", "blocks_per_episode"], None)
    if blocks_per_episode_val is None:
        raise ValueError("validation_v9 requires blocks_per_episode.")
    blocks_per_episode_i = int(blocks_per_episode_val)
    if blocks_per_episode_i < 1:
        raise ValueError("validation_v9 blocks_per_episode must be >= 1.")

    k_values = [int(x) for x in list(_flat_or_path(cfg, "k_values", ["phase_A", "k_values"], [0, 4, 8, 16, 32]))]
    if 0 not in k_values:
        raise ValueError("validation_v9.phase_A.k_values must include 0 baseline.")
    max_K = int(_flat_or_path(cfg, "max_K", ["phase_A", "max_K"], max(k_values)))
    if int(max(k_values)) != int(max_K):
        raise ValueError("validation_v9.phase_A.max_K must equal max(k_values).")
    if int(max_K) < 1:
        raise ValueError("validation_v9 max_K must be >= 1.")
    k_values = sorted(set(int(x) for x in k_values))

    segments_per_scene = int(_flat_or_path(cfg, "segments_per_scene", ["selection", "segments_per_scene"], 1))
    if segments_per_scene != 1:
        raise ValueError("P0 validation_v9 supports segments_per_scene=1 only.")
    segment_policy = str(_flat_or_path(cfg, "segment_policy", ["selection", "segment_policy"], "random_seeded"))
    episode_policy = str(_flat_or_path(cfg, "episode_policy", ["selection", "episode_policy"], "random_seeded"))
    blocks_per_segment = int(_flat_or_path(cfg, "blocks_per_segment", ["selection", "blocks_per_segment"], 4))
    block_policy = str(_flat_or_path(cfg, "block_policy", ["selection", "block_policy"], "random_without_replacement"))
    source_frame_policy = str(_flat_or_path(cfg, "source_frame_policy", ["selection", "source_frame_policy"], "middle_in_keyframe"))
    nearby_enable = bool(_flat_or_path(cfg, "nearby_enable", ["phase_A", "nearby", "enable"], True))
    nearby_frames_per_block = int(
        _flat_or_path(cfg, "nearby_frames_per_block", ["phase_A", "nearby", "frames_per_block"], 1)
    )
    nearby_policy = str(
        _flat_or_path(cfg, "nearby_policy", ["phase_A", "nearby", "policy"], "adjacent_then_random_same_keyframe")
    )
    nearby_same_keyframe_only = bool(
        _flat_or_path(cfg, "nearby_same_keyframe_only", ["phase_A", "nearby", "same_keyframe_only"], True)
    )
    nearby_camera_policy = str(_flat_or_path(cfg, "nearby_camera_policy", ["phase_A", "nearby", "camera_policy"], "all_cams"))
    nearby_adjacent_radius = int(_flat_or_path(cfg, "nearby_adjacent_radius", ["phase_A", "nearby", "adjacent_radius"], 1))
    nearby_random_fill = bool(_flat_or_path(cfg, "nearby_random_fill", ["phase_A", "nearby", "random_fill"], True))
    if not nearby_same_keyframe_only:
        raise ValueError("validation_v9.phase_A.nearby.same_keyframe_only must be true")
    if nearby_camera_policy != "all_cams":
        raise ValueError("validation_v9 P0 supports nearby camera_policy=all_cams only")

    specs: List[ValidationBlockSpecV9] = []
    for scene_id in eval_ids:
        segment_id = choose_segment_for_scene(
            dataset,
            int(scene_id),
            seed=int(seed),
            policy=str(segment_policy),
        )
        if segment_id is None:
            continue
        sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
        kfs = [int(x) for x in list(sidx.keyframe_indices)]
        effective_blocks_per_episode = min(int(blocks_per_episode_i), int(len(kfs)))
        start = choose_episode_start(
            num_keyframes=len(kfs),
            blocks_per_episode=int(blocks_per_episode_i),
            seed=int(seed),
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            policy=str(episode_policy),
        )
        if start is None:
            msg = (
                "not enough keyframes for validation episode: "
                f"scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"num_keyframes={len(kfs)} blocks_per_episode={int(blocks_per_episode_i)}"
            )
            if bool(fail_fast):
                raise ValueError(msg)
            continue
        keyframe_window = [int(x) for x in kfs[int(start) : int(start) + int(effective_blocks_per_episode)]]
        if len(keyframe_window) < int(effective_blocks_per_episode):
            if bool(fail_fast):
                raise ValueError("not enough keyframes for validation episode.")
            continue
        frame_chain = [
            choose_source_frame(
                list(sidx.keyframe_to_frames[int(kf)]),
                seed=int(seed),
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                keyframe_idx=int(kf),
                policy=str(source_frame_policy),
            )
            for kf in keyframe_window
        ]
        block_indices = choose_blocks(
            blocks_per_episode=int(effective_blocks_per_episode),
            n=int(blocks_per_segment),
            seed=int(seed),
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            policy=str(block_policy),
        )
        for block_idx in block_indices:
            source_kf = int(keyframe_window[int(block_idx)])
            source_frame = int(frame_chain[int(block_idx)])
            evidence_refs = _all_cams(source_frame, int(sidx.num_cams))
            block_loss_refs = _all_cams(source_frame, int(sidx.num_cams))
            nearby_frames: List[int] = []
            if bool(nearby_enable):
                nearby_frames = sample_nearby_frames_for_validation(
                    sidx=sidx,
                    source_keyframe_idx=int(source_kf),
                    source_frame=int(source_frame),
                    existing_loss_frames=[int(source_frame)],
                    num_frames=int(nearby_frames_per_block),
                    policy=str(nearby_policy),
                    adjacent_radius=int(nearby_adjacent_radius),
                    random_fill=bool(nearby_random_fill),
                    seed=int(seed),
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                )
            nearby_loss_refs = [
                ref
                for frame_idx in nearby_frames
                for ref in _all_cams(int(frame_idx), int(sidx.num_cams))
            ]
            if set(nearby_loss_refs) & set(evidence_refs):
                raise ValueError("nearby_loss_refs must not overlap evidence_refs.")
            specs.append(
                ValidationBlockSpecV9(
                    phase="phase_A_block_local_unroll",
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    segment_choice_rank=0,
                    episode_start_keyframe_pos=int(start),
                    keyframe_window=[int(x) for x in keyframe_window],
                    frame_chain=[int(x) for x in frame_chain],
                    block_idx=int(block_idx),
                    source_keyframe_idx=int(source_kf),
                    source_frame_idx=int(source_frame),
                    evidence_refs=[tuple(x) for x in evidence_refs],
                    block_loss_refs=[tuple(x) for x in block_loss_refs],
                    nearby_loss_refs=[tuple(x) for x in nearby_loss_refs],
                    num_cams=int(sidx.num_cams),
                    meta={
                        "selection_seed": int(seed),
                        "segment_policy": str(segment_policy),
                        "episode_policy": str(episode_policy),
                        "block_policy": str(block_policy),
                        "source_frame_policy": str(source_frame_policy),
                        "requested_blocks_per_episode": int(blocks_per_episode_i),
                        "effective_blocks_per_episode": int(effective_blocks_per_episode),
                    },
                )
            )
    if not specs and bool(fail_fast):
        raise ValueError("validation_v9 enabled but no valid block specs can be built.")
    return ValidationPlanV9(
        scheduler_version="v9",
        phase="phase_A_block_local_unroll",
        seed=int(seed),
        eval_scene_ids=[int(x) for x in eval_ids],
        blocks_per_episode=int(blocks_per_episode_i),
        k_values=[int(x) for x in k_values],
        max_K=int(max_K),
        block_specs=specs,
        request_meta={
            "scheduler_version": "v9",
            "validation_version": "v9",
            "validation_phase": "phase_A_block_local_unroll",
            "validation_k_values": [int(x) for x in k_values],
            "validation_max_K": int(max_K),
            "num_block_specs": int(len(specs)),
        },
    )


def materialize_validation_v9_batch(
    dataset: Any,
    rollout: ViewSetRolloutBatchV9,
    *,
    include_test: bool = False,
) -> Dict[str, Any]:
    if not hasattr(dataset, "_assemble_segment_batch_from_v9_request"):
        raise ValueError("validation_v9 requires dataset._assemble_segment_batch_from_v9_request")
    validate_phase_a_rollout_plan_v9(rollout, dataset=dataset)
    return dataset._assemble_segment_batch_from_v9_request(
        scene_id=int(rollout.scene_id),
        segment_id=int(rollout.segment_id),
        v9_plan=rollout,
        include_test=bool(include_test),
    )


__all__ = [
    "ValidationBlockSpecV9",
    "ValidationPlanV9",
    "build_request_meta_v9_for_validation",
    "build_validation_plan_v9",
    "choose_blocks",
    "choose_episode_start",
    "choose_segment_for_scene",
    "choose_source_frame",
    "make_phase_a_eval_rollout_batch",
    "materialize_validation_v9_batch",
    "sample_nearby_frames_for_validation",
    "validate_phase_a_rollout_plan_v9",
]
