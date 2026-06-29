from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0 import ContextPack, DeltaPack, EventPack, LocalGSState
from models.streetforward.stage6_0.phase_a_losses import delta_regularization

from .history_ema import IForwardResidualPack


class IForwardStage6Bridge:
    """Private low-level bridge into the Stage6 V4 measurement/render stack.

    IForward does not inherit the Stage6 trainer and never calls Stage6 phase
    forward/train paths. This adapter exposes only the reusable primitives:
    node-state initialization, V4 observation, event construction, updater and
    render supervision.
    """

    def __init__(self, runtime: Any):
        self.runtime = runtime

    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.runtime, "stage6_hidden_dim", 48))

    @property
    def event_dim(self) -> int:
        return int(getattr(self.runtime, "stage6_event_dim", 48))

    @property
    def current_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_block_mask_policy", "non_sky_non_egocar"))

    @property
    def nearby_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_nearby_mask_policy", "non_sky_non_egocar"))

    def get_or_init_node_states(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        return self.runtime._get_or_init_node_states_bg_rigid_distant(batch)

    def make_local_state(
        self,
        *,
        batch: Dict[str, Any],
    ) -> Tuple[LocalGSState, NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        node_state_bg, node_state_rigid, node_state_distant = self.get_or_init_node_states(batch)
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=int(self.hidden_dim),
        )
        return local_state, node_state_bg, node_state_distant, node_state_rigid

    def sync_local_state_template_from_batch(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
    ) -> Tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        node_state_bg, node_state_rigid, node_state_distant = self.get_or_init_node_states(batch)
        if local_state.rigid is None:
            if node_state_rigid is not None:
                raise ValueError("IForward carried local state has no rigid branch but batch node state has rigid branch.")
            return node_state_bg, node_state_distant, node_state_rigid
        if node_state_rigid is None:
            raise ValueError("IForward carried local state has rigid branch but batch node state has no rigid branch.")
        if int(local_state.rigid.means.shape[0]) != int(node_state_rigid.means.shape[0]):
            raise ValueError(
                "IForward carried rigid row count does not match current batch node state: "
                f"local={int(local_state.rigid.means.shape[0])} node={int(node_state_rigid.means.shape[0])}"
            )
        template = getattr(local_state, "rigid_template", None)
        if template is not None and tuple(template.point_ids.shape) == tuple(node_state_rigid.point_ids.shape):
            if not torch.equal(template.point_ids.to(device=node_state_rigid.point_ids.device), node_state_rigid.point_ids):
                raise ValueError("IForward carried rigid point ids do not match current batch node state.")
        local_state.rigid_template = node_state_rigid.detach_clone()
        return node_state_bg, node_state_distant, node_state_rigid

    def observe(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
        biggs_state: Optional[Any] = None,
        biggs_parent_runtime: Optional[Any] = None,
        biggs_scene_id: Optional[int] = None,
        biggs_segment_id: Optional[int] = None,
        biggs_episode_id: Optional[int] = None,
        parent_optimizer_state: Optional[Any] = None,
        visit_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.runtime._observe_v4_measurement(
            local_state=local_state,
            batch=batch,
            source_indices=[int(x) for x in source_indices],
            source_frame_idx=int(source_frame_idx),
            biggs_state=biggs_state,
            biggs_parent_runtime=biggs_parent_runtime,
            biggs_scene_id=biggs_scene_id,
            biggs_segment_id=biggs_segment_id,
            biggs_episode_id=biggs_episode_id,
            parent_optimizer_state=parent_optimizer_state,
            visit_meta=visit_meta,
        )

    def update_biggs_parent_runtime(
        self,
        *,
        runtime: Any,
        old_local_state: LocalGSState,
        new_local_state: LocalGSState,
    ) -> Any:
        updater = getattr(self.runtime, "_stage2_0_update_parent_runtime", None)
        if not callable(updater):
            return runtime
        return updater(
            runtime=runtime,
            old_local_state=old_local_state,
            new_local_state=new_local_state,
        )

    def build_stage2_1_parent_inputs(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Dict[str, Any]:
        builder = getattr(self.runtime, "_build_stage2_1_parent_inputs_from_measurement", None)
        if not callable(builder):
            raise RuntimeError("Stage2_1 requires runtime._build_stage2_1_parent_inputs_from_measurement.")
        return builder(local_state=local_state, measurement=measurement)

    def decode_stage2_1_biggs_child_event(
        self,
        *,
        parent_event: EventPack,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> EventPack:
        decoder = getattr(self.runtime, "_decode_stage2_1_biggs_child_event", None)
        if not callable(decoder):
            raise RuntimeError("Stage2_1 requires runtime._decode_stage2_1_biggs_child_event.")
        return decoder(parent_event=parent_event, local_state=local_state, measurement=measurement)

    def observe_planning(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
        biggs_state: Optional[Any] = None,
        biggs_scene_id: Optional[int] = None,
        biggs_segment_id: Optional[int] = None,
        biggs_episode_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        old_grad_mode = getattr(self.runtime, "stage6_source_evidence_grad_mode", None)
        had_grad_mode = hasattr(self.runtime, "stage6_source_evidence_grad_mode")
        if had_grad_mode:
            setattr(self.runtime, "stage6_source_evidence_grad_mode", "no_grad_v4")
        try:
            with torch.no_grad():
                return self.observe(
                    local_state=local_state,
                    batch=batch,
                    source_indices=source_indices,
                    source_frame_idx=int(source_frame_idx),
                    biggs_state=biggs_state,
                    biggs_scene_id=biggs_scene_id,
                    biggs_segment_id=biggs_segment_id,
                    biggs_episode_id=biggs_episode_id,
                )
        finally:
            if had_grad_mode:
                setattr(self.runtime, "stage6_source_evidence_grad_mode", old_grad_mode)

    def build_event(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> EventPack:
        return self.runtime._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)

    def stage6_aabb(self, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        getter = getattr(self.runtime, "_stage6_aabb", None)
        if callable(getter):
            return getter(ref)
        lo = ref.detach().amin(dim=0) - 1.0 if int(ref.numel()) > 0 else ref.new_full((3,), -1.0)
        hi = ref.detach().amax(dim=0) + 1.0 if int(ref.numel()) > 0 else ref.new_full((3,), 1.0)
        return lo.to(device=ref.device, dtype=ref.dtype), hi.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _branch_delta_norms(delta: Any, *, branch: str) -> Dict[str, float]:
        if delta is None:
            return {}
        out: Dict[str, float] = {}
        for name, value in (
            ("means", getattr(delta, "means", None)),
            ("scale", getattr(delta, "scales_log", None)),
            ("opacity", getattr(delta, "opacity_logit", None)),
            ("sh", getattr(delta, "sh", None)),
        ):
            if torch.is_tensor(value) and int(value.numel()) > 0:
                out[f"delta/{name}_norm_{branch}"] = float(value.detach().reshape(int(value.shape[0]), -1).norm(dim=-1).mean().item())
            else:
                out[f"delta/{name}_norm_{branch}"] = 0.0
        return out

    def apply_update(
        self,
        *,
        local_state: LocalGSState,
        event: EventPack,
        ctx_memory: Optional[ContextPack],
    ) -> Tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        local_state_out, delta, aux = self.runtime._apply_event_update(local_state=local_state, event=event, ctx_vsm=ctx_memory)
        aux.update(self._ctx_adapter_stats(event=event, ctx_memory=ctx_memory))
        aux.update(self._branch_delta_norms(delta.bg, branch="bg"))
        aux.update(self._branch_delta_norms(delta.distant, branch="distant"))
        aux.update(self._branch_delta_norms(delta.rigid, branch="rigid"))
        return local_state_out, delta, aux

    def _ctx_adapter_stats(self, *, event: EventPack, ctx_memory: Optional[ContextPack]) -> Dict[str, float]:
        adapter = getattr(getattr(self.runtime, "stage6_posterior_updater", None), "vsm_ctx_adapter", None)
        out: Dict[str, float] = {}
        if adapter is None or ctx_memory is None:
            return out
        with torch.no_grad():
            for branch, ctx, event_x in (
                ("bg", getattr(ctx_memory, "ctx_bg", None), getattr(event, "event_bg", None)),
                ("distant", getattr(ctx_memory, "ctx_distant", None), getattr(event, "event_distant", None)),
                ("rigid", getattr(ctx_memory, "ctx_rigid", None), getattr(event, "event_rigid", None)),
            ):
                if ctx is None or not torch.is_tensor(ctx) or int(ctx.numel()) == 0:
                    continue
                y = adapter(ctx.detach())
                input_norm = ctx.detach().norm(dim=-1).mean()
                output_norm = y.detach().norm(dim=-1).mean()
                event_norm = (
                    event_x.detach().norm(dim=-1).mean()
                    if torch.is_tensor(event_x) and int(event_x.numel()) > 0
                    else output_norm.new_tensor(0.0)
                )
                out[f"vsm_ctx_adapter/input_norm_{branch}"] = float(input_norm.item())
                out[f"vsm_ctx_adapter/output_norm_{branch}"] = float(output_norm.item())
                out[f"vsm_ctx_adapter/output_event_ratio_{branch}"] = float((output_norm / event_norm.clamp_min(1.0e-6)).item())
                out[f"adapter_output_norm_{branch}"] = float(y.detach().norm(dim=-1).mean().item())
        return out

    def predict_delta(
        self,
        *,
        local_state: LocalGSState,
        event: EventPack,
        ctx_memory: Optional[ContextPack],
    ) -> Tuple[DeltaPack, Dict[str, Any]]:
        _ = local_state
        updater = getattr(self.runtime, "stage6_posterior_updater", None)
        if not callable(updater):
            raise RuntimeError("IForward v3 requires runtime.stage6_posterior_updater to predict deltas.")
        delta, aux = updater(
            event=event,
            ctx_current=None,
            ctx_vsm=ctx_memory,
            appearance_detail=getattr(event, "appearance_detail", None),
        )
        out = {**dict(getattr(event, "aux", {}) or {}), **dict(aux or {})}
        out.update(self._ctx_adapter_stats(event=event, ctx_memory=ctx_memory))
        out.update(self._branch_delta_norms(delta.bg, branch="bg_raw"))
        out.update(self._branch_delta_norms(delta.distant, branch="distant_raw"))
        out.update(self._branch_delta_norms(delta.rigid, branch="rigid_raw"))
        return delta, out

    def apply_branch_scope_event_rows(self, delta: DeltaPack) -> DeltaPack:
        apply_scope = getattr(self.runtime, "_apply_branch_scope", None)
        if not callable(apply_scope):
            raise RuntimeError("IForward v3 requires runtime._apply_branch_scope.")
        return apply_scope(delta)

    def expand_rigid_delta(
        self,
        *,
        delta: DeltaPack,
        event: EventPack,
        local_state: LocalGSState,
    ) -> DeltaPack:
        if delta.rigid is None or local_state.rigid is None:
            return delta
        route = getattr(event, "route", None)
        rows = getattr(route, "S", None) if route is not None else None
        if rows is None:
            raise RuntimeError("IForward v3 rigid delta requires event.route.S before expansion.")
        expand = getattr(self.runtime, "_expand_branch_delta", None)
        if not callable(expand):
            raise RuntimeError("IForward v3 requires runtime._expand_branch_delta for rigid rows.")
        return DeltaPack(
            bg=delta.bg,
            distant=delta.distant,
            rigid=expand(delta.rigid, indices=rows, total=int(local_state.rigid.means.shape[0])),
            aux=delta.aux,
        )

    def apply_delta_only(self, *, local_state: LocalGSState, delta: DeltaPack) -> LocalGSState:
        next_state = local_state.apply_delta(delta)
        constrain = getattr(self.runtime, "_constrain_local_state_after_delta", None)
        if callable(constrain):
            next_state = constrain(next_state)
        return next_state

    @staticmethod
    def _spatial_hw_from_image(image: torch.Tensor) -> Tuple[int, int]:
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() < 2:
            raise ValueError(f"IForward v3 residual image must have spatial dims, got {tuple(image.shape)}")
        return int(image.shape[0]), int(image.shape[1])

    @staticmethod
    def _render_params_to_gaussians(render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "means": render_params["means_r"],
            "scales": render_params["scales_r"],
            "quats": render_params["quats_r"],
            "opacities": render_params["opacities_r"],
            "colors": render_params["colors_r"],
        }

    def _source_masks(self, batch: Dict[str, Any]) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        getter = getattr(self.runtime, "_get_source_masks_from_batch", None)
        if callable(getter):
            return getter(batch)
        return batch.get("source_sky_masks"), batch.get("source_egocar_masks")

    def _valid_mask_for_source(self, *, image: torch.Tensor, sky_mask: Any, egocar_mask: Any) -> torch.Tensor:
        builder = getattr(self.runtime, "_build_source_pair_valid_mask", None)
        if callable(builder):
            return builder(source_images=[image], source_sky_masks=[sky_mask], source_egocar_masks=[egocar_mask])
        h, w = self._spatial_hw_from_image(image)
        return torch.ones((1, h, w), device=image.device, dtype=torch.float32)

    def _branch_render_params(self, branch: Any) -> Dict[str, torch.Tensor]:
        fn = getattr(self.runtime, "_branch_render_params", None)
        if not callable(fn):
            raise RuntimeError("IForward v3 residual history requires runtime._branch_render_params.")
        return fn(branch, detach=True)

    @staticmethod
    def _cat_render_params(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "means_r": torch.cat([p["means_r"] for p in parts], dim=0),
            "scales_r": torch.cat([p["scales_r"] for p in parts], dim=0),
            "quats_r": torch.cat([p["quats_r"] for p in parts], dim=0),
            "opacities_r": torch.cat([p["opacities_r"] for p in parts], dim=0),
            "colors_r": torch.cat([p["colors_r"] for p in parts], dim=0),
        }

    def _residual_render_params_for_frame(
        self,
        *,
        local_state: LocalGSState,
        frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor, int]:
        parts = [self._branch_render_params(local_state.bg)]
        num_bg = int(local_state.bg.means.shape[0])
        rigid_idx = local_state.bg.means.new_zeros((0,), dtype=torch.long)
        rigid_node = None
        local_rigid_node = getattr(self.runtime, "_local_rigid_node_state", None)
        if callable(local_rigid_node):
            rigid_node = local_rigid_node(local_state)
        if local_state.rigid is not None and rigid_node is not None:
            valid_fn = getattr(self.runtime, "_stage6_rigid_point_valid_mask", None)
            rigid_world_fn = getattr(self.runtime, "_rigid_local_to_world_render_params", None)
            if not callable(valid_fn) or not callable(rigid_world_fn):
                raise RuntimeError(
                    "IForward v3 residual history requires runtime._stage6_rigid_point_valid_mask "
                    "and runtime._rigid_local_to_world_render_params."
                )
            valid = valid_fn(rigid_node, int(frame_idx))
            rigid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if int(rigid_idx.numel()) > 0:
                rigid_local_all = self._branch_render_params(local_state.rigid)
                rigid_local = {k: v[rigid_idx] for k, v in rigid_local_all.items()}
                point_ids = rigid_node.point_ids[rigid_idx, 0]
                parts.append(rigid_world_fn(rigid_node, rigid_local, int(frame_idx), point_ids_subset=point_ids))
        num_rigid_active = int(rigid_idx.numel())
        if local_state.distant is not None:
            parts.append(self._branch_render_params(local_state.distant))
        return self._cat_render_params(parts), num_bg, rigid_idx, num_rigid_active

    @torch.no_grad()
    def compute_block_residual_history(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
    ) -> IForwardResidualPack:
        render_rgb = getattr(getattr(self.runtime, "alpha_t_extractor", None), "render_rgb_only", None)
        backproject = getattr(self.runtime, "_backproject_scene_features_multi_camera", None)
        if not callable(render_rgb) or not callable(backproject):
            raise RuntimeError(
                "IForward v3 residual history requires runtime.alpha_t_extractor.render_rgb_only "
                "and runtime._backproject_scene_features_multi_camera."
            )
        source_views = list(batch.get("source_views") or [])
        source_images = list(batch.get("source_images") or [])
        if not source_views or not source_images:
            raise RuntimeError("IForward v3 residual history requires batch source_views/source_images.")
        if len(source_views) != len(source_images):
            raise RuntimeError("IForward v3 residual source_views/source_images length mismatch.")
        source_sky_masks, source_egocar_masks = self._source_masks(batch)
        source_refs = list((batch.get("request_meta") or {}).get("source_image_refs") or [])

        ref = local_state.bg.means
        num_bg = int(local_state.bg.means.shape[0])
        num_distant = int(local_state.distant.means.shape[0]) if local_state.distant is not None else 0
        num_rigid = int(local_state.rigid.means.shape[0]) if local_state.rigid is not None else 0
        support_bg_acc = ref.new_zeros((num_bg, 1))
        support_distant_acc = ref.new_zeros((num_distant, 1))
        support_rigid_acc = ref.new_zeros((num_rigid, 1))
        error_bg_num = ref.new_zeros((num_bg, 1))
        error_distant_num = ref.new_zeros((num_distant, 1))
        error_rigid_num = ref.new_zeros((num_rigid, 1))

        for raw_idx in [int(x) for x in source_indices]:
            if raw_idx < 0 or raw_idx >= len(source_views):
                raise RuntimeError(f"IForward v3 residual source index {raw_idx} out of range.")
            view = source_views[raw_idx]
            gt_image = source_images[raw_idx]
            if torch.is_tensor(gt_image) and gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            if not torch.is_tensor(gt_image):
                raise RuntimeError("IForward v3 residual source image must be a tensor.")
            gt_image = gt_image.to(device=ref.device, dtype=ref.dtype)
            h, w = self._spatial_hw_from_image(gt_image)
            frame_idx = int(source_refs[raw_idx][0]) if raw_idx < len(source_refs) else int(source_frame_idx)
            render_params, frame_num_bg, rigid_idx, num_rigid_active = self._residual_render_params_for_frame(
                local_state=local_state,
                frame_idx=int(frame_idx),
            )
            if int(frame_num_bg) != int(num_bg):
                raise RuntimeError("IForward v3 residual bg split mismatch.")
            gaussians_scene = self._render_params_to_gaussians(render_params)
            pred_rgb_l, _ = render_rgb(
                gaussians_scene,
                [view],
                int(h),
                int(w),
                return_acc=True,
                return_debug_stats=False,
            )
            residual = torch.abs(pred_rgb_l[0].to(device=ref.device, dtype=ref.dtype) - gt_image).mean(dim=-1, keepdim=True)
            sky = source_sky_masks[raw_idx] if source_sky_masks is not None and raw_idx < len(source_sky_masks) else None
            ego = source_egocar_masks[raw_idx] if source_egocar_masks is not None and raw_idx < len(source_egocar_masks) else None
            source_pair_valid_mask = self._valid_mask_for_source(image=gt_image, sky_mask=sky, egocar_mask=ego)
            error_all, acc_w_all = backproject(
                gaussians_scene=gaussians_scene,
                source_views=[view],
                features_2d=residual.unsqueeze(0),
                source_pair_valid_mask=source_pair_valid_mask,
                height=int(h),
                width=int(w),
                backprojector_override=getattr(self.runtime, "stage5_2_record_backprojector", None),
            )
            total = int(gaussians_scene["means"].shape[0])
            if error_all is None:
                error_all = ref.new_zeros((total, 1))
            if acc_w_all is None:
                acc_w_all = ref.new_zeros((total,))
            acc_bg = acc_w_all[:num_bg].to(dtype=ref.dtype).unsqueeze(-1)
            err_bg = error_all[:num_bg].to(dtype=ref.dtype)
            support_bg_acc = support_bg_acc + acc_bg
            error_bg_num = error_bg_num + err_bg * acc_bg

            offset = num_bg
            if num_rigid > 0 and num_rigid_active > 0:
                acc_rigid = acc_w_all[offset : offset + num_rigid_active].to(dtype=ref.dtype).unsqueeze(-1)
                err_rigid = error_all[offset : offset + num_rigid_active].to(dtype=ref.dtype)
                support_rigid_acc[rigid_idx] = support_rigid_acc[rigid_idx] + acc_rigid
                error_rigid_num[rigid_idx] = error_rigid_num[rigid_idx] + err_rigid * acc_rigid
            offset += num_rigid_active
            if num_distant > 0:
                acc_distant = acc_w_all[offset : offset + num_distant].to(dtype=ref.dtype).unsqueeze(-1)
                err_distant = error_all[offset : offset + num_distant].to(dtype=ref.dtype)
                support_distant_acc = support_distant_acc + acc_distant
                error_distant_num = error_distant_num + err_distant * acc_distant

        eps = 1.0e-6
        error_bg = torch.where(support_bg_acc > 0, error_bg_num / (support_bg_acc + eps), torch.zeros_like(error_bg_num))
        error_distant = (
            torch.where(
                support_distant_acc > 0,
                error_distant_num / (support_distant_acc + eps),
                torch.zeros_like(error_distant_num),
            )
            if num_distant > 0
            else None
        )
        error_rigid = (
            torch.where(
                support_rigid_acc > 0,
                error_rigid_num / (support_rigid_acc + eps),
                torch.zeros_like(error_rigid_num),
            )
            if num_rigid > 0
            else None
        )
        return IForwardResidualPack(
            error_bg=error_bg.detach(),
            support_bg=torch.log1p(support_bg_acc.clamp_min(0.0)).detach(),
            error_distant=None if error_distant is None else error_distant.detach(),
            support_distant=None if num_distant <= 0 else torch.log1p(support_distant_acc.clamp_min(0.0)).detach(),
            error_rigid=None if error_rigid is None else error_rigid.detach(),
            support_rigid=None if num_rigid <= 0 else torch.log1p(support_rigid_acc.clamp_min(0.0)).detach(),
        )

    def render_loss(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        target_indices: List[int],
        mask_policy: str,
        pred_rgbs_out: Optional[List[torch.Tensor]] = None,
        gt_images_out: Optional[List[torch.Tensor]] = None,
        return_per_ref_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.runtime._render_loss_for_indices(
            local_state=local_state,
            batch=batch,
            target_indices=[int(x) for x in target_indices],
            mask_policy=str(mask_policy),
            pred_rgbs_out=pred_rgbs_out,
            gt_images_out=gt_images_out,
            return_per_ref_loss=bool(return_per_ref_loss),
        )

    def history_probe_loss(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        target_indices: List[int],
        mask_policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=[int(x) for x in target_indices],
            mask_policy=str(mask_policy),
            pred_rgbs_out=None,
            gt_images_out=None,
        )

    def render_loss_for_targets(
        self,
        *,
        local_state: LocalGSState,
        ref_batch: Dict[str, Any],
        targets: List[Dict[str, Any]],
        mask_policy: str,
        pred_rgbs_out: Optional[List[torch.Tensor]] = None,
        gt_images_out: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not targets:
            ref = local_state.bg.means
            return ref.new_tensor(0.0), {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
        batch = dict(ref_batch)
        batch["targets"] = list(targets)
        return self.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=list(range(len(targets))),
            mask_policy=mask_policy,
            pred_rgbs_out=pred_rgbs_out,
            gt_images_out=gt_images_out,
        )

    def delta_regularization(
        self,
        delta: DeltaPack,
        *,
        local_state: LocalGSState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return delta_regularization(
            delta,
            weight=float(getattr(self.runtime, "stage6_delta_l2_weight", 1.0e-3)),
            local_state=local_state,
            opacity_delta_l2_weight=float(getattr(self.runtime, "stage6_opacity_delta_l2_weight", 0.0)),
            sh_delta_l2_weight=float(getattr(self.runtime, "stage6_sh_delta_l2_weight", 0.0)),
            scale_barrier_weight=float(getattr(self.runtime, "stage6_scale_barrier_weight", 0.0)),
            scale_log_min=float(getattr(self.runtime, "stage6_scale_log_min", -10.0)),
            scale_log_max=float(getattr(self.runtime, "stage6_scale_log_max", 4.0)),
        )

    def writeback_detached(
        self,
        *,
        local_state: LocalGSState,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
    ) -> None:
        local_state.writeback_detached(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
        )

    def reset_runtime_node_state(self) -> Dict[str, int]:
        before = {
            "bg": int(len(getattr(self.runtime, "node_states_bg", {}) or {})),
            "distant": int(len(getattr(self.runtime, "node_states_distant", {}) or {})),
            "rigid": int(len(getattr(self.runtime, "node_states_rigid", {}) or {})),
            "sky": int(len(getattr(self.runtime, "node_states_sky", {}) or {})),
        }
        reset = getattr(self.runtime, "reset_node_state", None)
        if callable(reset):
            reset()
        else:
            for name in ("node_states_bg", "node_states_distant", "node_states_rigid", "node_states_sky"):
                cache = getattr(self.runtime, name, None)
                if hasattr(cache, "clear"):
                    cache.clear()
            for name in ("h_cache_bg", "h_cache_distant", "h_cache_rigid", "h_cache_sky"):
                cache = getattr(self.runtime, name, None)
                if hasattr(cache, "clear"):
                    cache.clear()
        after = {
            "bg": int(len(getattr(self.runtime, "node_states_bg", {}) or {})),
            "distant": int(len(getattr(self.runtime, "node_states_distant", {}) or {})),
            "rigid": int(len(getattr(self.runtime, "node_states_rigid", {}) or {})),
            "sky": int(len(getattr(self.runtime, "node_states_sky", {}) or {})),
        }
        stats = {f"before_{k}": int(v) for k, v in before.items()}
        stats.update({f"after_{k}": int(v) for k, v in after.items()})
        return stats
