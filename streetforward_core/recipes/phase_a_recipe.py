from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0.local_gs_state import LocalGSState
from streetforward_core.data.resolvers.phase_a_resolver import PhaseABatchResolver
from streetforward_core.legacy.stage6_facade import (
    PhaseAEventBuilderAdapter,
    PosteriorUpdaterAdapter,
    Stage5NodeStateProviderAdapter,
    Stage5RendererAdapter,
    Stage5V4MeasurementAdapter,
    Stage6EventUpdateAdapter,
    Stage6LegacyFacade,
)
from models.streetforward.stage6_0.phase_a_losses import delta_regularization


@dataclass
class PhaseAForwardOutput:
    loss: torch.Tensor
    local_state: LocalGSState
    node_state_bg: NodeStateBackground
    node_state_distant: Optional[NodeStateDistant]
    node_state_rigid: Optional[NodeStateRigid]
    resolved: Any
    per_step: List[Dict[str, float]]
    pred_rgbs: List[torch.Tensor]
    gt_images: List[torch.Tensor]
    num_targets: int
    num_source_views: int

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "local_G": self.local_state,
            "node_state_bg": self.node_state_bg,
            "node_state_distant": self.node_state_distant,
            "node_state_rigid": self.node_state_rigid,
            "roles": self.resolved,
            "per_step": self.per_step,
            "num_targets": int(self.num_targets),
            "num_source_views": int(self.num_source_views),
            "pred_rgbs": list(self.pred_rgbs),
            "gt_images": list(self.gt_images),
        }


class PhaseARecipe(nn.Module):
    """Legacy-backed Phase A recipe.

    The algorithm boundary is explicit here, but the concrete adapters still
    call MinimalStreetForwardStage6_0 private helpers for parity.
    """

    def __init__(
        self,
        *,
        facade: Stage6LegacyFacade,
        resolver: Optional[PhaseABatchResolver] = None,
        node_state_provider: Optional[Stage5NodeStateProviderAdapter] = None,
        measurement: Optional[Stage5V4MeasurementAdapter] = None,
        event_builder: Optional[PhaseAEventBuilderAdapter] = None,
        posterior_updater: Optional[PosteriorUpdaterAdapter] = None,
        event_update: Optional[Stage6EventUpdateAdapter] = None,
        renderer: Optional[Stage5RendererAdapter] = None,
    ):
        super().__init__()
        self.facade = facade
        self.resolver = resolver if resolver is not None else PhaseABatchResolver()
        self.node_state_provider = (
            node_state_provider if node_state_provider is not None else Stage5NodeStateProviderAdapter(facade)
        )
        self.measurement = measurement if measurement is not None else Stage5V4MeasurementAdapter(facade)
        self.event_update = event_update
        self.event_builder = event_builder if event_builder is not None else PhaseAEventBuilderAdapter(facade)
        self.posterior_updater = posterior_updater if posterior_updater is not None else PosteriorUpdaterAdapter(facade)
        self.renderer = renderer if renderer is not None else Stage5RendererAdapter(facade)

    def forward(self, batch: Dict[str, Any]) -> PhaseAForwardOutput:
        resolved = self.resolver.resolve(batch)
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("Stage6 Phase A requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("Stage6 Phase A requires non-empty targets.")

        node_state_bg, node_state_rigid, node_state_distant = self.node_state_provider.get_or_init(batch)
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=int(self.facade.hidden_dim),
        )
        total_loss = local_state.bg.means.new_tensor(0.0)
        per_step: List[Dict[str, float]] = []
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        global_step = int(batch.get("global_step", 0) or 0)

        for k in range(int(resolved.inner_K)):
            step_plan = resolved.plan.steps[int(k)]
            source_frame_idx = int(step_plan.evidence_refs[0].frame_idx)
            measurement = self.measurement.observe(
                local_state=local_state,
                batch=batch,
                source_indices=list(resolved.evidence_source_indices_by_step[int(k)]),
                source_frame_idx=source_frame_idx,
            )
            if self.event_update is not None:
                local_state, delta, update_aux = self.event_update.encode_and_update(
                    local_state=local_state,
                    measurement=measurement,
                )
            else:
                event = self.event_builder.build_event(local_state=local_state, measurement=measurement)
                local_state, delta, update_aux = self.posterior_updater.apply_update(
                    local_state=local_state,
                    event=event,
                )
            final_step = int(k) == int(resolved.inner_K) - 1
            block_loss, block_stats = self.renderer.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=list(resolved.block_target_indices_by_step[int(k)]),
                mask_policy=self.facade.block_mask_policy,
                pred_rgbs_out=pred_rgbs if final_step else None,
                gt_images_out=gt_images if final_step else None,
            )
            nearby_loss, nearby_stats = self.renderer.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=list(resolved.nearby_target_indices_by_step[int(k)]),
                mask_policy=self.facade.nearby_mask_policy,
                pred_rgbs_out=pred_rgbs if final_step else None,
                gt_images_out=gt_images if final_step else None,
            )
            reg_loss, reg_stats = delta_regularization(
                delta,
                local_state=local_state,
                **self.facade.delta_regularization_kwargs(),
            )
            near_weight = self.facade.nearby_weight(global_step=global_step, k=int(k), K=int(resolved.inner_K))
            step_weight = float(self.facade.step_gamma) ** float(int(resolved.inner_K) - 1 - int(k))
            loss_k = step_weight * (float(self.facade.block_weight) * block_loss + near_weight * nearby_loss + reg_loss)
            if not torch.isfinite(loss_k).all():
                raise RuntimeError("Stage6 Phase A loss became NaN/Inf.")
            total_loss = total_loss + loss_k

            item: Dict[str, float] = {
                "k": float(k),
                "loss_block": float(block_loss.detach().item()),
                "loss_nearby": float(nearby_loss.detach().item()),
                "nearby_weight": float(near_weight),
                "block_valid_ratio": float(block_stats.get("valid_ratio", 0.0)),
                "nearby_valid_ratio": float(nearby_stats.get("valid_ratio", 0.0)),
                "block_skipped": float(block_stats.get("skipped_no_valid_pixels", 0.0)),
                "nearby_skipped": float(nearby_stats.get("skipped_no_valid_pixels", 0.0)),
                "block_metric_valid": float(block_stats.get("metric_valid", 0.0)),
                "nearby_metric_valid": float(nearby_stats.get("metric_valid", 0.0)),
                "block_num_metric_refs": float(block_stats.get("num_metric_refs", 0.0)),
                "nearby_num_metric_refs": float(nearby_stats.get("num_metric_refs", 0.0)),
                **{name: float(value) for name, value in reg_stats.items()},
                **{name: float(value) for name, value in update_aux.items() if isinstance(value, (int, float))},
            }
            for prefix, stats in (("block", block_stats), ("nearby", nearby_stats)):
                for metric_name in ("psnr", "ssim", "l1"):
                    value = stats.get(metric_name)
                    if value is None:
                        continue
                    value_f = float(value)
                    if math.isfinite(value_f):
                        item[f"{prefix}_{metric_name}"] = value_f
            per_step.append(item)

        return PhaseAForwardOutput(
            loss=total_loss,
            local_state=local_state,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            resolved=resolved,
            per_step=per_step,
            pred_rgbs=pred_rgbs,
            gt_images=gt_images,
            num_targets=len(batch.get("targets", [])),
            num_source_views=len(batch.get("source_views", [])),
        )
