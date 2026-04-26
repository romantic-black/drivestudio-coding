"""
Minimal StreetForward Stage 5.0:
- keep Stage4.6 routed rigid/no-sky semantics
- replace bg + rigid.S_in GRU inputs with point-preserving xCPE struct decoder outputs
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import torch

from models.streetforward.minimal_trainer_stage4_6 import BgRigidInGRUInputs, MinimalStreetForwardStage4_6, RigidRoute
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders import StructDecoderInput, StreetForwardXCPEDecoder, cat_param_dict


class MinimalStreetForwardStage5_0(MinimalStreetForwardStage4_6):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_stage5_0_config(config)
        super().__init__(config, device, **kwargs)
        self._init_stage5_0_struct_decoder(config)
        self._rebuild_optimizer_after_stage5_modules()

    def _validate_stage5_0_config(self, config) -> None:
        self._validate_stage4_6_config(config)
        model_cfg = self._require_key(config, "model", "config")
        stage = model_cfg.get("stage")
        if stage is not None and str(stage) != "5_0":
            raise ValueError("Stage5_0 requires model.stage='5_0' when model.stage is set.")

        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        if not bool(self._require_key(struct_cfg, "enable", "model.struct_decoder")):
            raise ValueError("Stage5_0 requires model.struct_decoder.enable=true.")
        if str(self._require_key(struct_cfg, "type", "model.struct_decoder")) != "xcpe":
            raise ValueError("Stage5_0 struct_decoder.type must be 'xcpe'.")
        if str(self._require_key(struct_cfg, "scope", "model.struct_decoder")) != "bg_rigid_in":
            raise ValueError("Stage5_0 struct_decoder.scope must be 'bg_rigid_in'.")
        if bool(self._require_key(struct_cfg, "include_distant", "model.struct_decoder")):
            raise ValueError("Stage5_0 struct_decoder.include_distant must be false.")
        if bool(self._require_key(struct_cfg, "include_rigid_out", "model.struct_decoder")):
            raise ValueError("Stage5_0 struct_decoder.include_rigid_out must be false.")
        if not bool(self._require_key(struct_cfg, "point_preserving", "model.struct_decoder")):
            raise ValueError("Stage5_0 struct_decoder.point_preserving must be true.")
        if bool(struct_cfg.get("clamp_grid_coord", False)):
            raise ValueError("Stage5_0 does not support struct_decoder.clamp_grid_coord=true.")
        sparse_backend = str(self._require_key(struct_cfg, "sparse_backend", "model.struct_decoder")).lower()
        if sparse_backend != "spconv":
            raise ValueError("Stage5_0 struct_decoder.sparse_backend must be 'spconv'.")

        output_role = str(self._require_key(struct_cfg, "output_role", "model.struct_decoder"))
        if output_role != "gru_input":
            raise ValueError("Stage5_0 struct_decoder.output_role must be 'gru_input'.")

        future_cfg = self._require_key(struct_cfg, "future", "model.struct_decoder")
        if bool(self._require_key(future_cfg, "allow_pooling", "model.struct_decoder.future")):
            raise ValueError("Stage5_0 struct_decoder.future.allow_pooling must be false.")

        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        if bool(token_cfg.get("use_hidden_state", False)):
            raise ValueError("Stage5_0 does not support struct_decoder.token.use_hidden_state=true.")
        if bool(token_cfg.get("use_anchor_rgb", False)):
            raise ValueError(
                "Stage5_0 does not support struct_decoder.token.use_anchor_rgb=true; "
                "set it to false to avoid silent config drift."
            )

    def _rebuild_optimizer_after_stage5_modules(self) -> None:
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(self.config.optimizer.get("lr")),
            eps=float(self.config.optimizer.get("eps")),
            weight_decay=float(self.config.optimizer.get("weight_decay")),
        )

    def _init_stage5_0_struct_decoder(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        xcpe_cfg = self._require_key(struct_cfg, "xcpe", "model.struct_decoder")

        feat_2d_channels_cfg = int(self._require_key(struct_cfg, "feat_2d_channels", "model.struct_decoder"))
        feat_2d_channels_model = int(self._require_key(model_cfg, "feat_2d_channels", "model"))
        if feat_2d_channels_cfg != feat_2d_channels_model:
            raise ValueError(
                "Stage5_0 struct_decoder.feat_2d_channels must match model.feat_2d_channels "
                f"({feat_2d_channels_model}), got {feat_2d_channels_cfg}."
            )

        output_dim_cfg = struct_cfg.get("output_dim", "auto")
        output_dim = self.fused_in_dim if str(output_dim_cfg) == "auto" else int(output_dim_cfg)
        if int(output_dim) != int(self.fused_in_dim):
            raise ValueError(
                "Stage5_0 struct_decoder.output_dim must match GRU input dim self.fused_in_dim "
                f"({self.fused_in_dim}), got {output_dim}."
            )

        self.struct_decoder = StreetForwardXCPEDecoder(
            feat_2d_channels=feat_2d_channels_cfg,
            out_channels=int(output_dim),
            param_dim=17,
            branch_embed_dim=int(self._require_key(struct_cfg, "branch_embed_dim", "model.struct_decoder")),
            support_embed_dim=int(self._require_key(struct_cfg, "support_embed_dim", "model.struct_decoder")),
            param_embed_dim=int(self._require_key(struct_cfg, "param_embed_dim", "model.struct_decoder")),
            channels=int(self._require_key(struct_cfg, "channels", "model.struct_decoder")),
            voxel_size=float(self._require_key(struct_cfg, "voxel_size", "model.struct_decoder")),
            num_layers=int(self._require_key(xcpe_cfg, "num_layers", "model.struct_decoder.xcpe")),
            kernel_size=int(self._require_key(xcpe_cfg, "kernel_size", "model.struct_decoder.xcpe")),
            residual_scale_init=float(self._require_key(xcpe_cfg, "residual_scale_init", "model.struct_decoder.xcpe")),
            sparse_backend=str(self._require_key(struct_cfg, "sparse_backend", "model.struct_decoder")),
            norm=str(xcpe_cfg.get("norm", "layernorm")),
            act=str(xcpe_cfg.get("act", "gelu")),
            use_2d_feat=bool(token_cfg.get("use_2d_feat", True)),
            use_support=bool(token_cfg.get("use_support", True)),
            use_branch_embed=bool(token_cfg.get("use_branch_embed", True)),
            use_param_embed=bool(token_cfg.get("use_param_embed", True)),
            zero_invalid_2d_feat=bool(token_cfg.get("zero_invalid_2d_feat", True)),
            clamp_grid_coord=bool(struct_cfg.get("clamp_grid_coord", False)),
        ).to(self.device)

        self.stage5_struct_enabled = True
        cfg_hash_src = self._to_hashable_struct_cfg(struct_cfg)
        self.struct_decoder_cfg_hash = hashlib.sha1(cfg_hash_src.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_hashable_struct_cfg(struct_cfg: Any) -> str:
        if isinstance(struct_cfg, dict):
            serializable = struct_cfg
        else:
            try:
                serializable = {k: struct_cfg.get(k) for k in struct_cfg.keys()}
            except Exception:
                serializable = str(struct_cfg)
        return json.dumps(serializable, sort_keys=True, default=str)

    def _build_struct_decoder_input_bg_rigid_in(
        self,
        *,
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_rigid_S: Optional[torch.Tensor],
    ) -> StructDecoderInput:
        num_bg = int(node_state_bg.means.shape[0])
        num_rigid_in = int(route.S_in.numel())

        feat_2d_parts = [feat_2d_bg]
        acc_w_parts = [acc_w_bg]
        coords_parts = [node_state_bg.means]

        branch_id_bg = torch.zeros((num_bg,), dtype=torch.long, device=self.device)
        branch_ids = [branch_id_bg]

        if num_rigid_in > 0:
            if node_state_rigid is None:
                raise RuntimeError("Stage5_0 got route.S_in > 0 but node_state_rigid is None.")
            if feat_2d_rigid_S is None or acc_w_rigid_S is None:
                raise RuntimeError("Stage5_0 expected rigid source 2D/support features for S_in.")

            rows_rigid_in_in_S = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_2d_rigid_in = feat_2d_rigid_S[rows_rigid_in_in_S]
            acc_w_rigid_in = acc_w_rigid_S[rows_rigid_in_in_S]
            coords_rigid_in = route.means_world_S[route.inside_mask_S]

            feat_2d_parts.append(feat_2d_rigid_in)
            acc_w_parts.append(acc_w_rigid_in)
            coords_parts.append(coords_rigid_in)
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=self.device))

        feat_2d = torch.cat(feat_2d_parts, dim=0)
        acc_w = torch.cat(acc_w_parts, dim=0)
        coords = torch.cat(coords_parts, dim=0)
        branch_id = torch.cat(branch_ids, dim=0)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        params_rigid_in = None
        if num_rigid_in > 0:
            params_rigid_in = self._build_rigid_params_for_embed_source_world(
                node_state_rigid,
                source_frame_idx,
                route.S_in,
            )
        params_struct = cat_param_dict(params_bg, params_rigid_in)

        return StructDecoderInput(
            feat_2d=feat_2d,
            acc_w=acc_w,
            coords=coords,
            branch_id=branch_id,
            params_for_embed=params_struct,
            split_bg=num_bg,
            split_rigid_in=num_rigid_in,
            meta={
                "support_threshold_bg": float(self.bg_src_backproject_support_min),
                "support_threshold_rigid": float(self.rigid_src_backproject_support_min),
            },
        )

    def _build_struct_batch_offsets(self, struct_in: StructDecoderInput) -> torch.Tensor:
        return torch.tensor([int(struct_in.coords.shape[0])], device=self.device, dtype=torch.long)

    def _compute_bg_rigid_in_gru_inputs(
        self,
        *,
        batch: Optional[Dict[str, Any]] = None,
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_rigid_S: Optional[torch.Tensor],
        node_state_distant: Optional[NodeStateDistant] = None,
        feat_2d_distant: Optional[torch.Tensor] = None,
        acc_w_distant: Optional[torch.Tensor] = None,
    ) -> BgRigidInGRUInputs:
        _ = batch
        _ = node_state_distant
        _ = feat_2d_distant
        _ = acc_w_distant
        struct_in = self._build_struct_decoder_input_bg_rigid_in(
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_bg=feat_2d_bg,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_bg=acc_w_bg,
            acc_w_rigid_S=acc_w_rigid_S,
        )
        batch_offsets = self._build_struct_batch_offsets(struct_in)
        struct_out = self.struct_decoder(
            struct_in,
            aabb_min=self.bbx_min,
            aabb_max=self.bbx_max,
            batch_offsets=batch_offsets,
        )

        feat_struct = struct_out.feat
        num_bg = int(struct_in.split_bg)
        num_rigid_in = int(struct_in.split_rigid_in)
        feat_bg_input = feat_struct[:num_bg]
        feat_rigid_in_input_all = feat_struct[num_bg : num_bg + num_rigid_in] if num_rigid_in > 0 else None

        aux = dict(struct_out.aux)
        aux.update(
            {
                "stage5_struct_enabled": 1.0,
                "stage5_struct_num_bg": float(num_bg),
                "stage5_struct_num_rigid_in": float(num_rigid_in),
                "stage5_bg_struct_feat_norm": float(feat_bg_input.norm(dim=-1).mean().item()) if num_bg > 0 else 0.0,
                "stage5_rigid_in_struct_feat_norm": (
                    float(feat_rigid_in_input_all.norm(dim=-1).mean().item()) if feat_rigid_in_input_all is not None else 0.0
                ),
            }
        )
        return BgRigidInGRUInputs(
            feat_bg_input=feat_bg_input,
            feat_rigid_in_input_all=feat_rigid_in_input_all,
            aux=aux,
        )

    @torch.no_grad()
    def demo_infer_step(
        self,
        batch: Dict[str, Any],
        *,
        scheduler_events: Optional[List[Dict[str, Any]]] = None,
        update_node_state: bool = True,
        update_hidden_state: bool = True,
    ) -> Dict[str, Any]:
        del scheduler_events
        prev_mode = self.training
        self.eval()
        out = self.forward(batch)
        if update_hidden_state and "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()
        if update_node_state:
            self._writeback_node_states_from_out(out)
        if prev_mode:
            self.train()
        loss_val = out.get("loss")
        return {
            "loss": loss_val.item() if torch.is_tensor(loss_val) else float(loss_val) if loss_val is not None else 0.0,
            "pred_rgbs": out.get("pred_rgbs"),
            "gt_images": out.get("gt_images"),
            "pred_rgb": out.get("pred_rgb"),
            "gt_image": out.get("gt_image"),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "num_bg_update": int(out.get("_num_bg_update", 0)),
            "num_distant_update": int(out.get("_num_distant_update", 0)),
            "num_rigid_update": int(out.get("_num_rigid_update", 0)),
        }


__all__ = ["MinimalStreetForwardStage5_0"]
