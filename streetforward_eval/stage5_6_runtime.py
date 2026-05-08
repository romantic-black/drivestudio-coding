from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .batch_builder import make_refs_for_frames
from .episode_builder import TestEpisodeSpec

ImageRef = Tuple[int, int]

logger = logging.getLogger(__name__)


def convert_batch_to_minimal_format(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from tools.train_minimal_streetforward_stage1_1 import (
        convert_batch_to_minimal_format as _convert_batch_to_minimal_format,
    )

    return _convert_batch_to_minimal_format(*args, **kwargs)


class BatchEvalOptimizerAdapter:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip_norm: float = 0.0,
        global_step: int = 0,
    ) -> None:
        self._optimizer = optimizer
        self.grad_clip_norm = float(grad_clip_norm)
        self.global_step = int(global_step)
        self.last_grad_norm = 0.0

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def state(self):
        return self._optimizer.state

    def zero_grad(self, *args: Any, **kwargs: Any):
        return self._optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state = self._optimizer.state_dict()
        state["_sf_global_step"] = int(self.global_step)
        state["_sf_last_grad_norm"] = float(self.last_grad_norm)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        state = dict(state_dict)
        self.global_step = int(state.pop("_sf_global_step", 0))
        self.last_grad_norm = float(state.pop("_sf_last_grad_norm", 0.0))
        return self._optimizer.load_state_dict(state)

    def step(self, *args: Any, **kwargs: Any):
        params = [p for group in self._optimizer.param_groups for p in group["params"]]
        if float(self.grad_clip_norm) > 0.0 and len(params) > 0:
            total = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=float(self.grad_clip_norm),
                error_if_nonfinite=True,
            )
            self.last_grad_norm = float(total.item() if torch.is_tensor(total) else total)
        else:
            sq = 0.0
            for p in params:
                if p.grad is None:
                    continue
                sq += float(p.grad.detach().float().pow(2).sum().item())
            self.last_grad_norm = float(sq**0.5)
        out = self._optimizer.step(*args, **kwargs)
        self.global_step += 1
        return out


_SEGMENT_FINETUNE_MAIN_PREFIXES = (
    "image_feature_extractor.residual",
    "image_feature_extractor.residual_unet",
    "image_feature_extractor.fusion",
    "image_feature_extractor.fusion_neck",
    "struct_decoder",
    "stage5_2_history_proj",
    "stage5_2_gate_branch_embed",
    "stage5_2_gate_mlp",
    "current_obs_",
)

_SEGMENT_FINETUNE_MAIN_TOKENS = (
    "offset_gru",
    "gru_update",
    "gru_candidate",
    "gru_reset",
    "gru_to_head",
    "mlp_offset",
    "mlp_conv",
    "mlp_opacity",
    "gaussion_decoder",
    "gaussian_decoder",
)


def is_stage5_6_error_predictor_param(name: str) -> bool:
    return (
        name.startswith("stage5_6_error_head")
        or name.startswith("err_splat_proj_bg")
        or name.startswith("err_splat_proj_distant")
        or name.startswith("err_splat_proj_rigid")
    )


def is_stage5_6_feedback_fuser_param(name: str) -> bool:
    return (
        name.startswith("stage5_6_bg_fuser")
        or name.startswith("stage5_6_distant_fuser")
        or name.startswith("stage5_6_rigid_fuser")
    )


def is_sky_param(name: str) -> bool:
    return name.startswith("sky_branch") or name.startswith("sky_model") or "_sky" in name or name.startswith("sky_")


def is_segment_finetune_main_param(name: str) -> bool:
    if any(name.startswith(prefix) for prefix in _SEGMENT_FINETUNE_MAIN_PREFIXES):
        return True
    return any(token in name for token in _SEGMENT_FINETUNE_MAIN_TOKENS)


def configure_segment_finetune_optimizer(
    model: Any,
    *,
    finetune_cfg: Dict[str, Any],
    start_step: int,
    log_prefix: str = "batcheval",
) -> None:
    train_feedback_fuser = bool(finetune_cfg.get("train_feedback_fuser", False))
    freeze_dino = bool(finetune_cfg.get("freeze_dino", True))
    if not bool(finetune_cfg.get("freeze_error_predictor", True)):
        logger.warning(
            "[%s] finetune.freeze_error_predictor=false is ignored; "
            "Stage5_6 error predictor is always frozen in segment_finetune_train.",
            log_prefix,
        )
    freeze_sky_branch = bool(finetune_cfg.get("freeze_sky_branch", True))
    lr = float(finetune_cfg.get("lr", 2.0e-5))
    weight_decay = float(finetune_cfg.get("weight_decay", 1.0e-5))
    grad_clip_norm = float(finetune_cfg.get("grad_clip_norm", 0.0))

    params: List[torch.nn.Parameter] = []
    names: List[str] = []
    frozen_preview: List[str] = []
    for name, param in model.named_parameters():
        trainable = bool(is_segment_finetune_main_param(str(name)))
        if is_stage5_6_error_predictor_param(str(name)):
            trainable = False
        elif is_stage5_6_feedback_fuser_param(str(name)):
            trainable = bool(train_feedback_fuser)
        elif name.startswith("image_feature_extractor.dino_adapter.backbone"):
            trainable = not bool(freeze_dino)
        elif freeze_sky_branch and is_sky_param(str(name)):
            trainable = False
        param.requires_grad_(bool(trainable))
        if trainable:
            params.append(param)
            names.append(str(name))
        elif len(frozen_preview) < 12:
            frozen_preview.append(str(name))

    if len(params) == 0:
        raise ValueError("segment_finetune_train selected zero trainable parameters")

    base_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for group in base_optimizer.param_groups:
        group["logical_name"] = "batch_eval_segment_finetune"
        group["param_names"] = list(names)
    model.optimizer = BatchEvalOptimizerAdapter(
        base_optimizer,
        grad_clip_norm=float(grad_clip_norm),
        global_step=int(start_step),
    )
    if hasattr(model, "lr_scheduler"):
        model.lr_scheduler = None
    setattr(model, "global_step", int(start_step))
    logger.info(
        "[%s] segment_finetune optimizer configured trainable_params=%d lr=%g weight_decay=%g "
        "train_feedback_fuser=%s freeze_dino=%s preview=%s",
        log_prefix,
        len(names),
        lr,
        weight_decay,
        train_feedback_fuser,
        freeze_dino,
        names[:12],
    )
    logger.info("[%s] segment_finetune frozen preview=%s", log_prefix, frozen_preview)


def iter_block_visit_order(
    *,
    num_blocks: int,
    steps_per_block: int,
    block_order: str,
    step_major_switch_interval_steps: int,
) -> List[int]:
    if int(num_blocks) < 1:
        return []
    if int(steps_per_block) < 1:
        raise ValueError("steps_per_input must be >= 1")
    order = str(block_order).strip()
    if order == "block_major":
        return [int(b) for b in range(int(num_blocks)) for _ in range(int(steps_per_block))]
    if order == "step_major":
        switch_every = int(step_major_switch_interval_steps)
        if switch_every < 1:
            raise ValueError("step_major_switch_interval_steps must be >= 1")
        out: List[int] = []
        for round_base in range(0, int(steps_per_block), int(switch_every)):
            chunk = int(min(int(switch_every), int(steps_per_block) - int(round_base)))
            for b in range(int(num_blocks)):
                out.extend([int(b)] * int(chunk))
        return out
    raise ValueError(f"unsupported block_order={block_order!r}, expected block_major/step_major")


def nearby_offsets_adjacent_non_input(spec: TestEpisodeSpec, input_offset: int) -> List[int]:
    input_set = set(int(x) for x in spec.input_offsets)
    valid_set = set(int(x) for x in spec.frame_offsets)
    out: List[int] = []
    for off in (int(input_offset) - 1, int(input_offset) + 1):
        if int(off) not in valid_set:
            continue
        if int(off) in input_set:
            continue
        out.append(int(off))
    return out


def expand_roles(frame_roles: List[str], camera_ids: List[int]) -> List[str]:
    return [str(role) for role in frame_roles for _ in camera_ids]


def update_camera_ids_for_spec(spec: TestEpisodeSpec, update_camera_ids: Optional[List[int]]) -> List[int]:
    if update_camera_ids is None:
        return [int(x) for x in spec.camera_ids]
    if len(update_camera_ids) == 0:
        raise ValueError("update_camera_ids must not be empty when provided")
    return [int(x) for x in update_camera_ids]


def nearby_frames_for_block(
    *,
    spec: TestEpisodeSpec,
    input_offset: int,
    existing_target_frames: List[int],
    input_frame_ids: List[int],
    nearby_policy: str,
    allow_partial_nearby: bool,
) -> List[int]:
    policy = str(nearby_policy).strip()
    if policy != "adjacent_non_input":
        raise ValueError(f"unsupported Stage5_6 nearby_policy={policy!r}, expected 'adjacent_non_input'")
    off_to_frame = {int(off): int(frame) for off, frame in zip(spec.frame_offsets, spec.frame_ids)}
    existing = set(int(x) for x in existing_target_frames) | set(int(x) for x in input_frame_ids)
    out: List[int] = []
    for off in nearby_offsets_adjacent_non_input(spec, int(input_offset)):
        if off not in off_to_frame:
            continue
        frame_id = int(off_to_frame[off])
        if frame_id in existing:
            continue
        if frame_id in out:
            continue
        out.append(int(frame_id))
    if len(out) < 2 and not bool(allow_partial_nearby):
        return []
    return out


def build_stage5_6_eval_train_batch(
    *,
    dataset: Any,
    spec: TestEpisodeSpec,
    block_idx: int,
    block_repeat_step: int,
    segment_local_step: int,
    visited_blocks: Set[int],
    device: torch.device,
    update_camera_ids: Optional[List[int]],
    protocol_name: str,
    steps_per_input: int,
    target_frame_policy: str,
    max_target_frames_including_source: Optional[int],
    nearby_policy: str,
    nearby_role_name: str,
    allow_partial_nearby: bool,
    block_order: str,
    step_major_switch_interval_steps: int,
) -> Dict[str, Any]:
    target_frame_policy = str(target_frame_policy).strip() or "visited_episode_frames"
    if target_frame_policy != "visited_episode_frames":
        target_frame_policy = "visited_episode_frames"
    camera_ids = update_camera_ids_for_spec(spec, update_camera_ids)
    input_frame_ids = [int(x) for x in spec.input_frame_ids]
    source_frame = int(input_frame_ids[int(block_idx)])
    limit = max_target_frames_including_source
    max_targets = len(input_frame_ids) if limit is None else max(int(limit), 1)
    prev_blocks = sorted([int(b) for b in visited_blocks if int(b) < int(block_idx)], reverse=True)
    next_blocks = sorted([int(b) for b in visited_blocks if int(b) > int(block_idx)])
    selected_blocks: List[int] = []
    for b in prev_blocks:
        if len(selected_blocks) >= int(max_targets - 1):
            break
        selected_blocks.append(int(b))
    for b in next_blocks:
        if len(selected_blocks) >= int(max_targets - 1):
            break
        selected_blocks.append(int(b))
    base_frames = [int(source_frame)] + [int(input_frame_ids[b]) for b in selected_blocks]
    base_roles = ["source"] + ["visited" for _ in base_frames[1:]]
    visited_frames_full = [int(source_frame)] + [int(input_frame_ids[b]) for b in sorted(int(x) for x in visited_blocks)]
    near_frames = nearby_frames_for_block(
        spec=spec,
        input_offset=int(spec.input_offsets[int(block_idx)]),
        existing_target_frames=[int(x) for x in (base_frames + visited_frames_full)],
        input_frame_ids=[int(x) for x in input_frame_ids],
        nearby_policy=str(nearby_policy),
        allow_partial_nearby=bool(allow_partial_nearby),
    )
    target_frames = [int(x) for x in base_frames] + [int(x) for x in near_frames]
    role_name = str(nearby_role_name)
    target_frame_roles = [str(x) for x in base_roles] + [role_name for _ in near_frames]

    source_image_refs = make_refs_for_frames(frame_ids=[source_frame], camera_ids=camera_ids)
    target_image_refs = make_refs_for_frames(frame_ids=target_frames, camera_ids=camera_ids)
    target_image_roles = expand_roles(target_frame_roles, camera_ids)
    if len(target_image_refs) != len(target_image_roles):
        raise ValueError(
            f"target_image_refs/target_image_roles mismatch: {len(target_image_refs)} vs {len(target_image_roles)}"
        )

    raw = dataset._assemble_segment_batch_from_image_refs(
        int(spec.scene_id),
        int(spec.segment_id),
        [(int(f), int(c)) for f, c in source_image_refs],
        [(int(f), int(c)) for f, c in target_image_refs],
        aux_image_refs=[],
        include_test=False,
        test_image_refs=None,
        enforce_target0_equals_source=True,
        target_ref_purpose="train",
    )
    request_meta = dict(raw.get("request_meta") or {})
    request_meta.update(
        {
            "eval_protocol": str(protocol_name),
            "batch_role": "update",
            "scheduler_version": "v8",
            "source_image_refs": [(int(f), int(c)) for f, c in source_image_refs],
            "target_image_refs": [(int(f), int(c)) for f, c in target_image_refs],
            "update_target_image_refs": [(int(f), int(c)) for f, c in target_image_refs],
            "target_frame_roles": [str(x) for x in target_frame_roles],
            "target_image_roles": [str(x) for x in target_image_roles],
            "observed_frame_ids": sorted(
                set(int(spec.input_frame_ids[int(b)]) for b in set(int(x) for x in visited_blocks) | {int(block_idx)})
            ),
            "source_frame_idx": int(source_frame),
            "near_random_frame_indices": [int(x) for x in near_frames],
            f"{role_name}_frame_indices": [int(x) for x in near_frames],
            "near_random_supervision_enable": True,
            "scheduler/near_random/enabled": 1.0,
            "scheduler/near_random/num_frames": float(len(near_frames)),
            "scheduler/near_random/skip_ratio": 0.0 if len(near_frames) > 0 else 1.0,
            "scheduler/near_random/num_candidate_frames_mean": float(len(near_frames)),
            "scheduler/near_random/sampled_blocks": 1.0 if len(near_frames) > 0 else 0.0,
        }
    )
    raw["request_meta"] = request_meta

    block_idx_global = int(spec.episode_idx) * int(max(len(spec.input_frame_ids), 1)) + int(block_idx)
    aligned = {
        "epoch_idx": 0,
        "global_step": int(segment_local_step),
        "scene_id": int(spec.scene_id),
        "segment_id": int(spec.segment_id),
        "segment_local_step": int(segment_local_step),
        "segment_local_u": int(segment_local_step),
        "segment_step_budget": int(len(spec.input_frame_ids) * int(steps_per_input)),
        "segment_budget_u": int(len(spec.input_frame_ids) * int(steps_per_input)),
        "block_idx_in_segment": int(block_idx_global),
        "block_idx_global": int(block_idx_global),
        "block_idx_in_episode": int(block_idx),
        "source_frame_idx": int(source_frame),
        "source_keyframe_idx": -1,
        "source_cam_idx": int(source_image_refs[0][1]),
        "source_image_ref": tuple(source_image_refs[0]),
        "source_image_refs": [(int(f), int(c)) for f, c in source_image_refs],
        "target_frame_indices": [int(x) for x in target_frames],
        "target_frame_roles": [str(x) for x in target_frame_roles],
        "target_image_refs": [(int(f), int(c)) for f, c in target_image_refs],
        "target_image_roles": [str(x) for x in target_image_roles],
        "near_random_frame_indices": [int(x) for x in near_frames],
        "target_policy": str(target_frame_policy),
        "visited_block_indices": sorted(int(x) for x in visited_blocks),
        "block_current_source_frame_indices": [int(x) for x in input_frame_ids],
        "U": 1,
        "K_u_nominal": int(steps_per_input),
        "K_u_effective": int(steps_per_input),
        "K_steps_effective": int(steps_per_input),
        "K_steps": int(steps_per_input),
        "R_steps": 0,
        "T_steps": int(steps_per_input),
        "episode_idx_global": int(spec.episode_idx),
        "block_repeat_step": int(block_repeat_step),
        "block_order": str(block_order),
        "step_major_switch_interval_steps": int(step_major_switch_interval_steps),
        "episode_step_cursor": int(segment_local_step),
        "scheduler_version": "v8",
    }
    raw["_scheduler_v4_aligned_info"] = dict(aligned)
    raw["_scheduler_v7_aligned_info"] = dict(aligned)
    raw["_scheduler_v8_aligned_info"] = dict(aligned)
    batch = convert_batch_to_minimal_format(
        raw,
        device=device,
        num_targets=len(target_image_refs),
        include_source_for_2d=True,
    )
    batch["request_meta"] = dict(batch.get("request_meta") or request_meta)
    batch["_scheduler_v4_aligned_info"] = dict(aligned)
    batch["_scheduler_v7_aligned_info"] = dict(aligned)
    batch["_scheduler_v8_aligned_info"] = dict(aligned)
    return batch


def stage5_6_runtime_policy(
    *,
    do_train: bool,
    update_hidden_state: bool,
    update_node_state: bool,
    reset_node_state_after_block: bool = False,
    force_eval_mode: bool = False,
):
    from models.streetforward.minimal_trainer_stage4_3 import RuntimePolicy

    return RuntimePolicy(
        do_backward=bool(do_train),
        do_optimizer_step=bool(do_train),
        update_hidden_cache=bool(update_hidden_state),
        writeback_node_state=bool(update_node_state),
        reset_node_state_after_block=bool(reset_node_state_after_block),
        force_eval_mode=bool(force_eval_mode),
    )


def stage5_6_scheduler_node_sync(*, segment_local_step: int) -> Dict[str, Any]:
    return {
        "U": 1,
        "segment_local_step": int(segment_local_step),
        "reset_after_block": False,
    }


def run_stage5_6_update_step(
    *,
    model: Any,
    update_batch: Dict[str, Any],
    mode: str,
    segment_local_step: int,
    update_hidden_state: bool,
    update_node_state: bool,
) -> Dict[str, Any]:
    sync = stage5_6_scheduler_node_sync(segment_local_step=int(segment_local_step))
    mode_norm = str(mode).strip()
    if mode_norm == "segment_finetune_train":
        return model.train_step(
            update_batch,
            step=None,
            profile_phase_timing=False,
            sync_cuda_timing=False,
            scheduler_node_sync=sync,
            runtime_policy=stage5_6_runtime_policy(
                do_train=True,
                update_hidden_state=bool(update_hidden_state),
                update_node_state=bool(update_node_state),
                force_eval_mode=False,
            ),
        )
    if mode_norm == "inference_only":
        with torch.no_grad():
            return model.inference_step_from_train_batch(
                update_batch,
                step=None,
                scheduler_node_sync=sync,
                runtime_policy=stage5_6_runtime_policy(
                    do_train=False,
                    update_hidden_state=bool(update_hidden_state),
                    update_node_state=bool(update_node_state),
                    force_eval_mode=False,
                ),
            )
    raise ValueError(f"unsupported Stage5_6 runtime mode={mode!r}")
