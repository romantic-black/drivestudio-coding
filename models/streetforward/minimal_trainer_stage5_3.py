"""
Minimal StreetForward Stage 5.3:
- full-branch routed inputs (near xCPE + far MLP)
- history memory + update gate interfaces
- block-exit record API
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.feature_extractors import DINOv2UNetFusionExtractor, FeatureBackprojector
from models.streetforward.minimal_trainer_stage4_6 import BgRigidInGRUInputs, MinimalStreetForwardStage4_6, RigidRoute
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders import (
    FarBranchMLPStructDecoder,
    RoutedNearFarStructDecoder,
    StructDecoderInput,
    StreetForwardXCPEDecoder,
    cat_param_dict,
)


@dataclass
class FullRoutedGRUInputs:
    feat_bg_input: torch.Tensor
    feat_distant_input: Optional[torch.Tensor]
    feat_rigid_in_input_all: Optional[torch.Tensor]
    feat_rigid_out_input_all: Optional[torch.Tensor]
    gate_bg: Optional[torch.Tensor]
    gate_distant: Optional[torch.Tensor]
    gate_rigid_in: Optional[torch.Tensor]
    gate_rigid_out: Optional[torch.Tensor]
    aux: Dict[str, Any]


class MinimalStreetForwardStage5_3(MinimalStreetForwardStage4_6):
    FAR_LOCAL_BRANCH_DISTANT = 0
    FAR_LOCAL_BRANCH_RIGID_OUT = 1

    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_stage5_3_config(config)
        super().__init__(config, device, **kwargs)
        self._init_stage5_3_modules(config)
        self._rebuild_optimizer_after_stage5_modules()

    def _validate_stage5_3_config(self, config) -> None:
        self._validate_stage4_6_config(config)
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model"))
        if stage != "5_3":
            raise ValueError("Stage5_3 requires model.stage='5_3'.")

        routed_cfg = self._require_key(model_cfg, "rigid_routed", "model")
        if str(self._require_key(routed_cfg, "inside_decoder", "model.rigid_routed")) != "bg":
            raise ValueError("Stage5_3 requires rigid_routed.inside_decoder=bg.")
        if str(self._require_key(routed_cfg, "outside_decoder", "model.rigid_routed")) != "distant":
            raise ValueError("Stage5_3 requires rigid_routed.outside_decoder=distant.")

        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        if not bool(self._require_key(struct_cfg, "enable", "model.struct_decoder")):
            raise ValueError("Stage5_3 requires model.struct_decoder.enable=true.")
        if str(self._require_key(struct_cfg, "type", "model.struct_decoder")) != "routed_near_far":
            raise ValueError("Stage5_3 requires model.struct_decoder.type='routed_near_far'.")
        if str(self._require_key(struct_cfg, "scope", "model.struct_decoder")) != "full_routed":
            raise ValueError("Stage5_3 requires model.struct_decoder.scope='full_routed'.")
        if str(self._require_key(struct_cfg, "output_role", "model.struct_decoder")) != "gru_input":
            raise ValueError("Stage5_3 requires model.struct_decoder.output_role='gru_input'.")
        if bool(self._require_key(struct_cfg, "point_preserving", "model.struct_decoder")) is not True:
            raise ValueError("Stage5_3 requires model.struct_decoder.point_preserving=true.")
        if bool(self._require_key(struct_cfg, "include_bg", "model.struct_decoder")) is not True:
            raise ValueError("Stage5_3 requires model.struct_decoder.include_bg=true.")
        if bool(self._require_key(struct_cfg, "include_distant", "model.struct_decoder")) is not True:
            raise ValueError("Stage5_3 requires model.struct_decoder.include_distant=true.")
        if bool(self._require_key(struct_cfg, "include_rigid_in", "model.struct_decoder")) is not True:
            raise ValueError("Stage5_3 requires model.struct_decoder.include_rigid_in=true.")
        if bool(self._require_key(struct_cfg, "include_rigid_out", "model.struct_decoder")) is not True:
            raise ValueError("Stage5_3 requires model.struct_decoder.include_rigid_out=true.")
        if bool(struct_cfg.get("clamp_grid_coord", False)):
            raise ValueError("Stage5_3 does not support clamp_grid_coord=true.")
        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        if bool(token_cfg.get("use_hidden_state", False)):
            raise ValueError("Stage5_3 requires struct_decoder.token.use_hidden_state=false.")
        if bool(token_cfg.get("use_anchor_rgb", False)):
            raise ValueError("Stage5_3 requires struct_decoder.token.use_anchor_rgb=false.")
        near_cfg = self._require_key(struct_cfg, "near", "model.struct_decoder")
        far_cfg = self._require_key(struct_cfg, "far", "model.struct_decoder")
        if str(self._require_key(near_cfg, "type", "model.struct_decoder.near")) != "xcpe":
            raise ValueError("Stage5_3 near decoder type must be xcpe.")
        if str(self._require_key(far_cfg, "type", "model.struct_decoder.far")) != "mlp":
            raise ValueError("Stage5_3 far decoder type must be mlp.")
        near_branches = list(self._require_key(near_cfg, "branches", "model.struct_decoder.near"))
        if [str(x) for x in near_branches] != ["bg", "rigid_in"]:
            raise ValueError("Stage5_3 requires near.branches == ['bg', 'rigid_in'].")
        far_branches = list(self._require_key(far_cfg, "branches", "model.struct_decoder.far"))
        if [str(x) for x in far_branches] != ["distant", "rigid_out"]:
            raise ValueError("Stage5_3 requires far.branches == ['distant', 'rigid_out'].")
        if bool(near_cfg.get("clamp_grid_coord", False)):
            raise ValueError("Stage5_3 near xCPE does not support clamp_grid_coord=true.")

        history_cfg = self._require_key(model_cfg, "history_memory", "model")
        if not bool(self._require_key(history_cfg, "enable", "model.history_memory")):
            raise ValueError("Stage5_3 requires history_memory.enable=true.")
        if str(self._require_key(history_cfg, "record_on", "model.history_memory")) != "block_exit":
            raise ValueError("Stage5_3 v1 requires history_memory.record_on=block_exit.")
        record_views = str(self._require_key(history_cfg, "record_views", "model.history_memory"))
        if record_views != "source_image_refs":
            raise ValueError("Stage5_3 requires history_memory.record_views=source_image_refs.")
        support_cfg = self._require_key(history_cfg, "support", "model.history_memory")
        residual_cfg = self._require_key(history_cfg, "residual", "model.history_memory")
        update_cfg = self._require_key(history_cfg, "update", "model.history_memory")
        _ = self._require_key(support_cfg, "ema_beta_visible", "model.history_memory.support")
        _ = self._require_key(support_cfg, "ema_beta_invisible", "model.history_memory.support")
        _ = self._require_key(residual_cfg, "error_beta", "model.history_memory.residual")
        _ = self._require_key(residual_cfg, "error_eps", "model.history_memory.residual")
        _ = self._require_key(update_cfg, "ema_beta", "model.history_memory.update")
        legacy_flat_history_keys = (
            "support_beta_visible",
            "support_beta_invisible",
            "error_beta",
            "update_norm_beta",
            "error_eps",
        )
        for legacy_key in legacy_flat_history_keys:
            if hasattr(history_cfg, "get") and history_cfg.get(legacy_key, None) is not None:
                raise ValueError(
                    "Stage5_3 no longer supports flat history_memory keys. "
                    "Use nested keys under history_memory.support/residual/update."
                )

        gate_cfg = self._require_key(model_cfg, "update_gate", "model")
        if not bool(self._require_key(gate_cfg, "enable", "model.update_gate")):
            raise ValueError("Stage5_3 requires update_gate.enable=true.")
        if not bool(self._require_key(gate_cfg, "bind_with_mask_update", "model.update_gate")):
            raise ValueError("Stage5_3 requires update_gate.bind_with_mask_update=true.")
        if not bool(self._require_key(gate_cfg, "require_initialized_in_input", "model.update_gate")):
            raise ValueError("Stage5_3 requires update_gate.require_initialized_in_input=true.")

        scheduler_cfg = self._require_key(config, "scheduler_v8", "config")
        episode_cfg = self._require_key(scheduler_cfg, "episode", "scheduler_v8")
        execution_cfg = self._require_key(scheduler_cfg, "execution", "scheduler_v8")
        if str(self._require_key(episode_cfg, "target_policy", "scheduler_v8.episode")) != "visited_episode_frames":
            raise ValueError("Stage5_3 requires scheduler_v8.episode.target_policy=visited_episode_frames.")
        if str(self._require_key(execution_cfg, "reset_policy", "scheduler_v8.execution")) != "episode_end":
            raise ValueError("Stage5_3 requires scheduler_v8.execution.reset_policy=episode_end.")

        feature_extractor_cfg = self._require_key(model_cfg, "feature_extractor", "model")
        feature_extractor_type = str(self._require_key(feature_extractor_cfg, "type", "model.feature_extractor")).strip().lower()
        if feature_extractor_type != "dinov2_unet_fusion":
            raise ValueError("Stage5_3 requires model.feature_extractor.type='dinov2_unet_fusion'.")

    def _rebuild_optimizer_after_stage5_modules(self) -> None:
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(self.config.optimizer.get("lr")),
            eps=float(self.config.optimizer.get("eps")),
            weight_decay=float(self.config.optimizer.get("weight_decay")),
        )

    def _init_stage5_3_modules(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        near_cfg = self._require_key(struct_cfg, "near", "model.struct_decoder")
        far_cfg = self._require_key(struct_cfg, "far", "model.struct_decoder")
        xcpe_cfg = self._require_key(near_cfg, "xcpe", "model.struct_decoder.near")
        gate_cfg = self._require_key(model_cfg, "update_gate", "model")
        history_cfg = self._require_key(model_cfg, "history_memory", "model")

        feat_2d_channels_cfg = int(self._require_key(struct_cfg, "feat_2d_channels", "model.struct_decoder"))
        feat_2d_channels_model = int(self._require_key(model_cfg, "feat_2d_channels", "model"))
        if feat_2d_channels_cfg != feat_2d_channels_model:
            raise ValueError(
                "Stage5_3 struct_decoder.feat_2d_channels must match model.feat_2d_channels "
                f"({feat_2d_channels_model}), got {feat_2d_channels_cfg}."
            )

        out_dim = int(self.fused_in_dim)
        hist_embed_dim = int(self._require_key(struct_cfg, "history_embed_dim", "model.struct_decoder"))
        near_decoder = StreetForwardXCPEDecoder(
            feat_2d_channels=feat_2d_channels_cfg,
            out_channels=out_dim,
            param_dim=17,
            branch_embed_dim=int(self._require_key(struct_cfg, "branch_embed_dim", "model.struct_decoder")),
            support_embed_dim=int(self._require_key(struct_cfg, "support_embed_dim", "model.struct_decoder")),
            param_embed_dim=int(self._require_key(struct_cfg, "param_embed_dim", "model.struct_decoder")),
            channels=int(self._require_key(near_cfg, "channels", "model.struct_decoder.near")),
            voxel_size=float(self._require_key(near_cfg, "voxel_size", "model.struct_decoder.near")),
            num_layers=int(self._require_key(xcpe_cfg, "num_layers", "model.struct_decoder.near.xcpe")),
            kernel_size=int(self._require_key(xcpe_cfg, "kernel_size", "model.struct_decoder.near.xcpe")),
            residual_scale_init=float(
                self._require_key(xcpe_cfg, "residual_scale_init", "model.struct_decoder.near.xcpe")
            ),
            sparse_backend=str(self._require_key(near_cfg, "sparse_backend", "model.struct_decoder.near")),
            norm=str(xcpe_cfg.get("norm", "layernorm")),
            act=str(xcpe_cfg.get("act", "gelu")),
            use_2d_feat=bool(token_cfg.get("use_2d_feat", True)),
            use_support=bool(token_cfg.get("use_support", True)),
            use_branch_embed=bool(token_cfg.get("use_branch_embed", True)),
            use_param_embed=bool(token_cfg.get("use_param_embed", True)),
            zero_invalid_2d_feat=bool(token_cfg.get("zero_invalid_2d_feat", True)),
            clamp_grid_coord=False,
        ).to(self.device)
        far_decoder = FarBranchMLPStructDecoder(
            feat_2d_channels=feat_2d_channels_cfg,
            out_channels=out_dim,
            param_dim=17,
            branch_embed_dim=int(self._require_key(struct_cfg, "branch_embed_dim", "model.struct_decoder")),
            support_embed_dim=int(self._require_key(struct_cfg, "support_embed_dim", "model.struct_decoder")),
            param_embed_dim=int(self._require_key(struct_cfg, "param_embed_dim", "model.struct_decoder")),
            channels=int(self._require_key(far_cfg, "channels", "model.struct_decoder.far")),
            hidden_dim=int(self._require_key(far_cfg, "hidden_dim", "model.struct_decoder.far")),
            num_layers=int(self._require_key(far_cfg, "num_layers", "model.struct_decoder.far")),
            norm=str(far_cfg.get("norm", "layernorm")),
            act=str(far_cfg.get("act", "gelu")),
            use_2d_feat=bool(token_cfg.get("use_2d_feat", True)),
            use_support=bool(token_cfg.get("use_support", True)),
            use_branch_embed=bool(token_cfg.get("use_branch_embed", True)),
            use_param_embed=bool(token_cfg.get("use_param_embed", True)),
            zero_invalid_2d_feat=bool(token_cfg.get("zero_invalid_2d_feat", True)),
            history_dim=int(hist_embed_dim),
        ).to(self.device)
        self.struct_decoder = RoutedNearFarStructDecoder(
            near_decoder=near_decoder,
            far_decoder=far_decoder,
        ).to(self.device)
        self.stage5_2_feat_2d_channels = int(feat_2d_channels_cfg)

        feature_extractor_cfg = self._require_key(model_cfg, "feature_extractor", "model")
        feature_extractor_type = str(self._require_key(feature_extractor_cfg, "type", "model.feature_extractor")).strip().lower()
        if feature_extractor_type != "dinov2_unet_fusion":
            raise ValueError(
                f"Stage5_3 unsupported model.feature_extractor.type={feature_extractor_type!r}. "
                "Only 'dinov2_unet_fusion' is supported."
            )
        dino_cfg = self._require_key(feature_extractor_cfg, "dino", "model.feature_extractor")
        residual_cfg = self._require_key(feature_extractor_cfg, "residual_unet", "model.feature_extractor")
        fusion_cfg = self._require_key(feature_extractor_cfg, "fusion", "model.feature_extractor")
        fusion_out_channels = int(self._require_key(fusion_cfg, "out_channels", "model.feature_extractor.fusion"))
        if fusion_out_channels != feat_2d_channels_model:
            raise ValueError(
                "Stage5_3 fusion out_channels must match model.feat_2d_channels "
                f"({feat_2d_channels_model}), got {fusion_out_channels}."
            )

        self.image_feature_extractor = DINOv2UNetFusionExtractor(
            dino_model_name=str(self._require_key(dino_cfg, "model_name", "model.feature_extractor.dino")),
            dino_pretrained=bool(self._require_key(dino_cfg, "pretrained", "model.feature_extractor.dino")),
            dino_weights_path=dino_cfg.get("weights_path", None),
            dino_freeze=bool(dino_cfg.get("freeze", True)),
            dino_out_channels=int(self._require_key(dino_cfg, "out_channels", "model.feature_extractor.dino")),
            dino_intermediate_layers=tuple(self._require_key(dino_cfg, "intermediate_layers", "model.feature_extractor.dino")),
            dino_pad_to_patch_multiple=int(
                self._require_key(dino_cfg, "pad_to_patch_multiple", "model.feature_extractor.dino")
            ),
            residual_in_channels=int(self._require_key(residual_cfg, "in_channels", "model.feature_extractor.residual_unet")),
            residual_feat_channels=int(
                self._require_key(residual_cfg, "feat_channels", "model.feature_extractor.residual_unet")
            ),
            residual_base_channels=int(
                self._require_key(residual_cfg, "base_channels", "model.feature_extractor.residual_unet")
            ),
            residual_feature_downscale=int(
                self._require_key(residual_cfg, "feature_downscale", "model.feature_extractor.residual_unet")
            ),
            residual_depth=int(self._require_key(residual_cfg, "depth", "model.feature_extractor.residual_unet")),
            residual_bilinear=bool(self._require_key(residual_cfg, "bilinear", "model.feature_extractor.residual_unet")),
            fusion_hidden_channels=int(
                self._require_key(fusion_cfg, "hidden_channels", "model.feature_extractor.fusion")
            ),
            fusion_out_channels=fusion_out_channels,
        ).to(self.device)

        gate_hidden_dim = int(self._require_key(gate_cfg, "hidden_dim", "model.update_gate"))
        branch_embed_dim = int(self._require_key(struct_cfg, "branch_embed_dim", "model.struct_decoder"))
        self.stage5_2_history_proj = nn.Sequential(
            nn.Linear(5, hist_embed_dim),
            nn.GELU(),
            nn.LayerNorm(hist_embed_dim),
        ).to(self.device)
        self.stage5_2_gate_branch_embed = nn.Embedding(4, branch_embed_dim).to(self.device)
        self.stage5_2_gate_mlp = nn.Sequential(
            nn.Linear(self.fused_in_dim + self.offset_gru_hidden_dim + self.param_embed_dim + hist_embed_dim + branch_embed_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        ).to(self.device)
        self.stage5_2_gate_min = float(self._require_key(gate_cfg, "min_gate", "model.update_gate"))
        self.stage5_2_gate_init_bias = float(self._require_key(gate_cfg, "init_bias", "model.update_gate"))
        with torch.no_grad():
            last = self.stage5_2_gate_mlp[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(self.stage5_2_gate_init_bias)

        support_cfg = self._require_key(history_cfg, "support", "model.history_memory")
        residual_cfg = self._require_key(history_cfg, "residual", "model.history_memory")
        update_cfg = self._require_key(history_cfg, "update", "model.history_memory")
        self.stage5_2_support_beta_visible = float(
            self._require_key(support_cfg, "ema_beta_visible", "model.history_memory.support")
        )
        self.stage5_2_support_beta_invisible = float(
            self._require_key(support_cfg, "ema_beta_invisible", "model.history_memory.support")
        )
        self.stage5_2_error_beta = float(self._require_key(residual_cfg, "error_beta", "model.history_memory.residual"))
        self.stage5_2_update_norm_beta = float(self._require_key(update_cfg, "ema_beta", "model.history_memory.update"))
        self.stage5_2_error_eps = float(self._require_key(residual_cfg, "error_eps", "model.history_memory.residual"))
        self.stage5_2_history_update_apply_in_eval = bool(update_cfg.get("apply_in_eval", False))
        self.stage5_2_record_views = str(self._require_key(history_cfg, "record_views", "model.history_memory"))
        self.stage5_2_record_backprojector = FeatureBackprojector(
            eps=getattr(self.feature_backprojector, "eps", 1e-8),
            weight_threshold=0.0,
        )
        self.stage5_2_bg_visible_min = float(self.bg_src_backproject_support_min)
        self.stage5_2_distant_visible_min = float(self.distant_src_backproject_support_min)
        self.stage5_2_rigid_visible_min = float(self.rigid_src_backproject_support_min)

        self.stage5_2_history_bg: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.stage5_2_history_distant: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.stage5_2_history_rigid: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.stage5_2_block_support_bg: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.stage5_2_block_support_distant: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.stage5_2_block_support_rigid: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._stage5_2_last_full_inputs: Optional[FullRoutedGRUInputs] = None

    @staticmethod
    def _build_struct_batch_offsets(struct_in: StructDecoderInput, *, device: torch.device) -> torch.Tensor:
        return torch.tensor([int(struct_in.coords.shape[0])], device=device, dtype=torch.long)

    def _get_or_init_history(
        self,
        cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
        key: Tuple[int, int],
        num_points: int,
    ) -> Dict[str, torch.Tensor]:
        cur = cache.get(key)
        if cur is not None and int(cur["support_ema"].shape[0]) == int(num_points):
            return cur
        z = torch.zeros((int(num_points), 1), dtype=torch.float32, device=self.device)
        cur = {
            "support_ema": z.clone(),
            "error_ema": z.clone(),
            "update_norm_ema": z.clone(),
            "initialized": z.clone(),
        }
        cache[key] = cur
        return cur

    def _get_or_init_block_support_acc(
        self,
        cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
        key: Tuple[int, int],
        num_points: int,
    ) -> Dict[str, torch.Tensor]:
        cur = cache.get(key)
        if cur is not None and int(cur["sum"].shape[0]) == int(num_points):
            return cur
        z = torch.zeros((int(num_points), 1), dtype=torch.float32, device=self.device)
        cur = {"sum": z.clone(), "count": z.clone()}
        cache[key] = cur
        return cur

    def _accumulate_support_before_update(
        self,
        *,
        key: Tuple[int, int],
        num_bg_total: int,
        num_distant_total: int,
        num_rigid_total: int,
        route: RigidRoute,
        acc_w_bg: torch.Tensor,
        acc_w_distant: Optional[torch.Tensor],
        acc_w_rigid_S: Optional[torch.Tensor],
    ) -> None:
        support_bg_cur = torch.log1p(acc_w_bg.detach()).unsqueeze(-1)
        acc_bg = self._get_or_init_block_support_acc(self.stage5_2_block_support_bg, key, num_bg_total)
        acc_bg["sum"] = acc_bg["sum"] + support_bg_cur
        acc_bg["count"] = acc_bg["count"] + torch.ones_like(support_bg_cur)

        if num_distant_total > 0:
            if acc_w_distant is None:
                raise RuntimeError("Stage5_3 support accumulation requires acc_w_distant for active distant branch.")
            support_distant_cur = torch.log1p(acc_w_distant.detach()).unsqueeze(-1)
            acc_distant = self._get_or_init_block_support_acc(
                self.stage5_2_block_support_distant,
                key,
                num_distant_total,
            )
            acc_distant["sum"] = acc_distant["sum"] + support_distant_cur
            acc_distant["count"] = acc_distant["count"] + torch.ones_like(support_distant_cur)

        if num_rigid_total > 0:
            acc_rigid = self._get_or_init_block_support_acc(self.stage5_2_block_support_rigid, key, num_rigid_total)
            support_rigid_full = torch.zeros((num_rigid_total, 1), dtype=torch.float32, device=self.device)
            count_rigid_full = torch.zeros((num_rigid_total, 1), dtype=torch.float32, device=self.device)
            if int(route.S.numel()) > 0:
                if acc_w_rigid_S is None:
                    raise RuntimeError("Stage5_3 support accumulation requires acc_w_rigid_S for active rigid branch.")
                if int(acc_w_rigid_S.shape[0]) != int(route.S.shape[0]):
                    raise RuntimeError("Stage5_3 support accumulation mismatch: len(acc_w_rigid_S) != len(route.S).")
                support_rigid_S = torch.log1p(acc_w_rigid_S.detach()).unsqueeze(-1)
                support_rigid_full[route.S] = support_rigid_S
                count_rigid_full[route.S] = 1.0
            acc_rigid["sum"] = acc_rigid["sum"] + support_rigid_full
            acc_rigid["count"] = acc_rigid["count"] + count_rigid_full

    def _apply_support_ema_update(
        self,
        *,
        history: Dict[str, torch.Tensor],
        support_cur: torch.Tensor,
        support_min: float,
    ) -> None:
        sv = float(self.stage5_2_support_beta_visible)
        si = float(self.stage5_2_support_beta_invisible)
        vis = (support_cur.squeeze(-1) > float(support_min)).unsqueeze(-1).to(dtype=history["support_ema"].dtype)
        history["support_ema"] = torch.where(
            vis > 0,
            sv * history["support_ema"] + (1.0 - sv) * support_cur.to(dtype=history["support_ema"].dtype),
            si * history["support_ema"] + (1.0 - si) * support_cur.to(dtype=history["support_ema"].dtype),
        )
        history["initialized"] = torch.maximum(history["initialized"], vis.to(dtype=history["initialized"].dtype))

    def _commit_block_support_to_history(
        self,
        *,
        key: Tuple[int, int],
        history_bg: Dict[str, torch.Tensor],
        history_distant: Dict[str, torch.Tensor],
        history_rigid: Dict[str, torch.Tensor],
    ) -> None:
        acc_bg = self.stage5_2_block_support_bg.get(key)
        if acc_bg is not None and int(acc_bg["sum"].shape[0]) == int(history_bg["support_ema"].shape[0]):
            support_bg = acc_bg["sum"] / acc_bg["count"].clamp(min=1.0)
            self._apply_support_ema_update(
                history=history_bg,
                support_cur=support_bg,
                support_min=float(self.stage5_2_bg_visible_min),
            )

        acc_distant = self.stage5_2_block_support_distant.get(key)
        if acc_distant is not None and int(acc_distant["sum"].shape[0]) == int(history_distant["support_ema"].shape[0]):
            support_distant = acc_distant["sum"] / acc_distant["count"].clamp(min=1.0)
            self._apply_support_ema_update(
                history=history_distant,
                support_cur=support_distant,
                support_min=float(self.stage5_2_distant_visible_min),
            )

        acc_rigid = self.stage5_2_block_support_rigid.get(key)
        if acc_rigid is not None and int(acc_rigid["sum"].shape[0]) == int(history_rigid["support_ema"].shape[0]):
            support_rigid = acc_rigid["sum"] / acc_rigid["count"].clamp(min=1.0)
            self._apply_support_ema_update(
                history=history_rigid,
                support_cur=support_rigid,
                support_min=float(self.stage5_2_rigid_visible_min),
            )

    def _clear_block_support_acc(self, key: Tuple[int, int]) -> None:
        self.stage5_2_block_support_bg.pop(key, None)
        self.stage5_2_block_support_distant.pop(key, None)
        self.stage5_2_block_support_rigid.pop(key, None)

    def _compute_gate(
        self,
        *,
        feat: Optional[torch.Tensor],
        h_old: Optional[torch.Tensor],
        params_for_embed: Optional[Dict[str, torch.Tensor]],
        history: Optional[Dict[str, torch.Tensor]],
        acc_w: Optional[torch.Tensor],
        support_min: float,
        branch_id: int,
    ) -> Optional[torch.Tensor]:
        if feat is None or h_old is None or params_for_embed is None or history is None or acc_w is None:
            return None
        if int(feat.shape[0]) == 0:
            return feat.new_zeros((0, 1))
        param_vec = self._normalize_params_for_embed(params_for_embed)
        param_embed = self.param_embed_norm(self.mlp_params_embed(param_vec))
        history_embed = self._build_history_embed(
            history=history,
            acc_w=acc_w,
            support_min=support_min,
            dtype=feat.dtype,
        )
        branch_embed = self.stage5_2_gate_branch_embed(
            torch.full((feat.shape[0],), int(branch_id), device=feat.device, dtype=torch.long)
        )
        gate_logits = self.stage5_2_gate_mlp(torch.cat([feat, h_old, param_embed, history_embed, branch_embed], dim=-1))
        gate = torch.sigmoid(gate_logits)
        return self.stage5_2_gate_min + (1.0 - self.stage5_2_gate_min) * gate

    def _build_history_embed(
        self,
        *,
        history: Dict[str, torch.Tensor],
        acc_w: torch.Tensor,
        support_min: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        visible_now = (acc_w > float(support_min)).to(dtype=dtype).unsqueeze(-1)
        history_raw = torch.cat(
            [
                history["support_ema"].to(dtype=dtype),
                history["error_ema"].to(dtype=dtype),
                history["update_norm_ema"].to(dtype=dtype),
                history["initialized"].to(dtype=dtype),
                visible_now,
            ],
            dim=-1,
        )
        return self.stage5_2_history_proj(history_raw)

    def _build_struct_decoder_input_near(
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
        branch_ids = [torch.zeros((num_bg,), dtype=torch.long, device=self.device)]

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        params_rigid_in = None
        if num_rigid_in > 0:
            if node_state_rigid is None or feat_2d_rigid_S is None or acc_w_rigid_S is None:
                raise RuntimeError("Stage5_3 near input requires rigid source tensors when S_in > 0.")
            rows_rigid_in_in_S = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_2d_parts.append(feat_2d_rigid_S[rows_rigid_in_in_S])
            acc_w_parts.append(acc_w_rigid_S[rows_rigid_in_in_S])
            coords_parts.append(route.means_world_S[route.inside_mask_S])
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=self.device))
            params_rigid_in = self._build_rigid_params_for_embed_source_world(node_state_rigid, source_frame_idx, route.S_in)

        return StructDecoderInput(
            feat_2d=torch.cat(feat_2d_parts, dim=0),
            acc_w=torch.cat(acc_w_parts, dim=0),
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=cat_param_dict(params_bg, params_rigid_in),
            split_bg=num_bg,
            split_rigid_in=num_rigid_in,
            meta={
                "support_threshold_bg": float(self.bg_src_backproject_support_min),
                "support_threshold_rigid": float(self.rigid_src_backproject_support_min),
            },
        )

    def _build_struct_decoder_input_far(
        self,
        *,
        source_frame_idx: int,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_distant: Optional[torch.Tensor],
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_distant: Optional[torch.Tensor],
        acc_w_rigid_S: Optional[torch.Tensor],
        history_embed: Optional[torch.Tensor] = None,
    ) -> StructDecoderInput:
        num_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_rigid_out = int(route.S_out.numel())
        feat_2d_parts = []
        acc_w_parts = []
        coords_parts = []
        branch_ids = []
        # Note: far decoder branch ids are far-local, not global gate branch ids.
        # far-local mapping: 0=distant, 1=rigid_out.

        params_distant = None
        if num_distant > 0:
            if feat_2d_distant is None or acc_w_distant is None or node_state_distant is None:
                raise RuntimeError("Stage5_3 far input expected distant 2D/support tensors.")
            feat_2d_parts.append(feat_2d_distant)
            acc_w_parts.append(acc_w_distant)
            coords_parts.append(node_state_distant.means)
            branch_ids.append(
                torch.full(
                    (num_distant,),
                    self.FAR_LOCAL_BRANCH_DISTANT,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")

        params_rigid_out = None
        if num_rigid_out > 0:
            if node_state_rigid is None or feat_2d_rigid_S is None or acc_w_rigid_S is None:
                raise RuntimeError("Stage5_3 far input expected rigid source tensors for S_out.")
            rows_rigid_out_in_S = torch.nonzero(~route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_2d_parts.append(feat_2d_rigid_S[rows_rigid_out_in_S])
            acc_w_parts.append(acc_w_rigid_S[rows_rigid_out_in_S])
            coords_parts.append(route.means_world_S[~route.inside_mask_S])
            branch_ids.append(
                torch.full(
                    (num_rigid_out,),
                    self.FAR_LOCAL_BRANCH_RIGID_OUT,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            params_rigid_out = self._build_rigid_params_for_embed_source_world(node_state_rigid, source_frame_idx, route.S_out)

        if len(feat_2d_parts) == 0:
            return StructDecoderInput(
                feat_2d=torch.zeros((0, int(self.stage5_2_feat_2d_channels)), device=self.device),
                acc_w=torch.zeros((0,), device=self.device),
                coords=torch.zeros((0, 3), device=self.device),
                branch_id=torch.zeros((0,), dtype=torch.long, device=self.device),
                params_for_embed={
                    "means": torch.zeros((0, 3), device=self.device),
                    "quats": torch.zeros((0, 4), device=self.device),
                    "scales_log": torch.zeros((0, 3), device=self.device),
                    "opacity_logit": torch.zeros((0, 1), device=self.device),
                    "sh_dc": torch.zeros((0, 3), device=self.device),
                    "sh_rest": torch.zeros((0, max(self.num_sh_bases - 1, 0), 3), device=self.device),
                },
                split_bg=0,
                split_rigid_in=0,
            )

        params_for_embed = params_distant
        if params_for_embed is None:
            params_for_embed = params_rigid_out
        elif params_rigid_out is not None:
            params_for_embed = cat_param_dict(params_for_embed, params_rigid_out)
        if params_for_embed is None:
            raise RuntimeError("Stage5_3 internal error: empty params_for_embed for far input.")

        meta: Dict[str, Any] = {
            "support_threshold_distant": float(self.distant_src_backproject_support_min),
            "support_threshold_rigid_out": float(self.rigid_src_backproject_support_min),
            "branch_id_space": "far_local",
            "branch_0": "distant",
            "branch_1": "rigid_out",
        }
        if history_embed is not None:
            meta["history_embed"] = history_embed

        return StructDecoderInput(
            feat_2d=torch.cat(feat_2d_parts, dim=0),
            acc_w=torch.cat(acc_w_parts, dim=0),
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=params_for_embed,
            split_bg=num_distant,
            split_rigid_in=num_rigid_out,
            meta=meta,
        )

    def _compute_full_routed_gru_inputs(
        self,
        *,
        batch: Dict[str, Any],
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_distant: Optional[torch.Tensor],
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_distant: Optional[torch.Tensor],
        acc_w_rigid_S: Optional[torch.Tensor],
    ) -> FullRoutedGRUInputs:
        key = self._batch_key(batch)
        num_bg_total = int(node_state_bg.means.shape[0])
        num_distant_total = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_rigid_total = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        history_bg = self._get_or_init_history(self.stage5_2_history_bg, key, num_bg_total)
        history_distant = self._get_or_init_history(self.stage5_2_history_distant, key, num_distant_total)
        history_rigid = self._get_or_init_history(self.stage5_2_history_rigid, key, num_rigid_total)
        self._accumulate_support_before_update(
            key=key,
            num_bg_total=num_bg_total,
            num_distant_total=num_distant_total,
            num_rigid_total=num_rigid_total,
            route=route,
            acc_w_bg=acc_w_bg,
            acc_w_distant=acc_w_distant,
            acc_w_rigid_S=acc_w_rigid_S,
        )

        near_in = self._build_struct_decoder_input_near(
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_bg=feat_2d_bg,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_bg=acc_w_bg,
            acc_w_rigid_S=acc_w_rigid_S,
        )
        near_out = self.struct_decoder.decode_near(
            near_in,
            aabb_min=self.bbx_min,
            aabb_max=self.bbx_max,
            batch_offsets=self._build_struct_batch_offsets(near_in, device=self.device),
        )
        num_bg = int(near_in.split_bg)
        num_rigid_in = int(near_in.split_rigid_in)
        feat_bg_input = near_out.feat[:num_bg]
        feat_rigid_in_input_all = near_out.feat[num_bg : num_bg + num_rigid_in] if num_rigid_in > 0 else None

        far_history_embed = None
        if num_distant_total + int(route.S_out.numel()) > 0:
            far_hist_parts = []
            if num_distant_total > 0:
                if acc_w_distant is None:
                    raise RuntimeError("Stage5_3 far history embed requires acc_w_distant when distant branch is active.")
                far_hist_parts.append(
                    self._build_history_embed(
                        history=history_distant,
                        acc_w=acc_w_distant,
                        support_min=self.stage5_2_distant_visible_min,
                        dtype=feat_2d_distant.dtype if feat_2d_distant is not None else node_state_bg.means.dtype,
                    )
                )
            num_rigid_out_total = int(route.S_out.numel())
            if num_rigid_out_total > 0:
                if acc_w_rigid_S is None:
                    raise RuntimeError("Stage5_3 far history embed requires acc_w_rigid_S when rigid_out branch is active.")
                rows_rigid_out_in_S = torch.nonzero(~route.inside_mask_S, as_tuple=False).squeeze(1)
                far_hist_parts.append(
                    self._build_history_embed(
                        history={
                            "support_ema": history_rigid["support_ema"][route.S_out],
                            "error_ema": history_rigid["error_ema"][route.S_out],
                            "update_norm_ema": history_rigid["update_norm_ema"][route.S_out],
                            "initialized": history_rigid["initialized"][route.S_out],
                        },
                        acc_w=acc_w_rigid_S[rows_rigid_out_in_S],
                        support_min=self.stage5_2_rigid_visible_min,
                        dtype=feat_2d_rigid_S.dtype if feat_2d_rigid_S is not None else node_state_bg.means.dtype,
                    )
                )
            if len(far_hist_parts) > 0:
                far_history_embed = torch.cat(far_hist_parts, dim=0)

        far_in = self._build_struct_decoder_input_far(
            source_frame_idx=source_frame_idx,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_distant=feat_2d_distant,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_distant=acc_w_distant,
            acc_w_rigid_S=acc_w_rigid_S,
            history_embed=far_history_embed,
        )
        feat_distant_input = None
        feat_rigid_out_input_all = None
        far_aux: Dict[str, Any] = {}
        if int(far_in.coords.shape[0]) > 0:
            far_out = self.struct_decoder.decode_far(
                far_in,
                aabb_min=self.bbx_min,
                aabb_max=self.bbx_max,
                batch_offsets=self._build_struct_batch_offsets(far_in, device=self.device),
            )
            n0 = int(far_in.split_bg)
            n1 = int(far_in.split_rigid_in)
            feat_distant_input = far_out.feat[:n0] if n0 > 0 else None
            feat_rigid_out_input_all = far_out.feat[n0 : n0 + n1] if n1 > 0 else None
            far_aux = dict(far_out.aux)
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, num_bg, node_state_bg, "bg")
        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        gate_bg = self._compute_gate(
            feat=feat_bg_input,
            h_old=h_old_bg,
            params_for_embed=params_bg,
            history=history_bg,
            acc_w=acc_w_bg,
            support_min=self.stage5_2_bg_visible_min,
            branch_id=0,
        )

        gate_distant = None
        if feat_distant_input is not None and node_state_distant is not None and acc_w_distant is not None:
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant,
                key,
                int(node_state_distant.means.shape[0]),
                node_state_distant,
                "distant",
            )
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            gate_distant = self._compute_gate(
                feat=feat_distant_input,
                h_old=h_old_distant,
                params_for_embed=params_distant,
                history=history_distant,
                acc_w=acc_w_distant,
                support_min=self.stage5_2_distant_visible_min,
                branch_id=2,
            )

        gate_rigid_in = None
        gate_rigid_out = None
        if node_state_rigid is not None:
            h_old_rigid = self._get_or_init_hidden(
                self.h_cache_rigid,
                key,
                int(node_state_rigid.means.shape[0]),
                node_state_rigid,
                "rigid",
            )
            if feat_rigid_in_input_all is not None and route.S_in.numel() > 0:
                params_rigid_in = self._build_rigid_params_for_embed_source_world(node_state_rigid, source_frame_idx, route.S_in)
                rows_rigid_in_in_S = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
                acc_w_rigid_in = acc_w_rigid_S[rows_rigid_in_in_S] if acc_w_rigid_S is not None else None
                gate_rigid_in = self._compute_gate(
                    feat=feat_rigid_in_input_all,
                    h_old=h_old_rigid[route.S_in],
                    params_for_embed=params_rigid_in,
                    history={
                        "support_ema": history_rigid["support_ema"][route.S_in],
                        "error_ema": history_rigid["error_ema"][route.S_in],
                        "update_norm_ema": history_rigid["update_norm_ema"][route.S_in],
                        "initialized": history_rigid["initialized"][route.S_in],
                    },
                    acc_w=acc_w_rigid_in,
                    support_min=self.stage5_2_rigid_visible_min,
                    branch_id=1,
                )
            if feat_rigid_out_input_all is not None and route.S_out.numel() > 0:
                params_rigid_out = self._build_rigid_params_for_embed_source_world(node_state_rigid, source_frame_idx, route.S_out)
                rows_rigid_out_in_S = torch.nonzero(~route.inside_mask_S, as_tuple=False).squeeze(1)
                acc_w_rigid_out = acc_w_rigid_S[rows_rigid_out_in_S] if acc_w_rigid_S is not None else None
                gate_rigid_out = self._compute_gate(
                    feat=feat_rigid_out_input_all,
                    h_old=h_old_rigid[route.S_out],
                    params_for_embed=params_rigid_out,
                    history={
                        "support_ema": history_rigid["support_ema"][route.S_out],
                        "error_ema": history_rigid["error_ema"][route.S_out],
                        "update_norm_ema": history_rigid["update_norm_ema"][route.S_out],
                        "initialized": history_rigid["initialized"][route.S_out],
                    },
                    acc_w=acc_w_rigid_out,
                    support_min=self.stage5_2_rigid_visible_min,
                    branch_id=3,
                )

        aux = dict(near_out.aux)
        aux.update(far_aux)
        aux.update(
            {
                "stage5_2_near_num_bg": float(num_bg),
                "stage5_2_near_num_rigid_in": float(num_rigid_in),
                "stage5_2_far_num_distant": float(int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0),
                "stage5_2_far_num_rigid_out": float(int(route.S_out.numel())),
            }
        )
        return FullRoutedGRUInputs(
            feat_bg_input=feat_bg_input,
            feat_distant_input=feat_distant_input,
            feat_rigid_in_input_all=feat_rigid_in_input_all,
            feat_rigid_out_input_all=feat_rigid_out_input_all,
            gate_bg=gate_bg,
            gate_distant=gate_distant,
            gate_rigid_in=gate_rigid_in,
            gate_rigid_out=gate_rigid_out,
            aux=aux,
        )

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
        if batch is None:
            raise ValueError("Stage5_3 requires batch in _compute_bg_rigid_in_gru_inputs.")
        full = self._compute_full_routed_gru_inputs(
            batch=batch,
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_bg=feat_2d_bg,
            feat_2d_distant=feat_2d_distant,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_bg=acc_w_bg,
            acc_w_distant=acc_w_distant,
            acc_w_rigid_S=acc_w_rigid_S,
        )
        self._stage5_2_last_full_inputs = full
        return BgRigidInGRUInputs(
            feat_bg_input=full.feat_bg_input,
            feat_rigid_in_input_all=full.feat_rigid_in_input_all,
            aux=dict(full.aux),
        )

    @staticmethod
    def _apply_update_gate(
        offsets: Dict[str, torch.Tensor],
        *,
        h_old: torch.Tensor,
        h_candidate: torch.Tensor,
        gate: Optional[torch.Tensor],
        mask_update: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if gate is None:
            return offsets, h_candidate, None
        dtype = h_old.dtype
        device = h_old.device
        eff = gate.to(device=device, dtype=dtype)
        if mask_update is not None:
            eff = eff * mask_update.to(device=device, dtype=dtype).unsqueeze(-1)
        offsets_g = dict(offsets)
        quat_identity = torch.zeros_like(offsets_g["offset_quat"])
        quat_identity[:, 0] = 1.0
        offsets_g["offset_pos"] = offsets_g["offset_pos"] * eff
        offsets_g["offset_scales"] = offsets_g["offset_scales"] * eff
        offsets_g["offset_opacity"] = offsets_g["offset_opacity"] * eff
        offsets_g["offset_sh"] = offsets_g["offset_sh"] * eff
        quat_mix = quat_identity * (1.0 - eff.expand_as(offsets_g["offset_quat"])) + offsets_g["offset_quat"] * eff.expand_as(offsets_g["offset_quat"])
        offsets_g["offset_quat"] = quat_mix / torch.clamp(torch.norm(quat_mix, dim=-1, keepdim=True), min=1e-8)
        h_new = (1.0 - eff) * h_old + eff * h_candidate
        if mask_update is not None:
            m = mask_update.bool()
            if bool((~m).any().item()):
                if not torch.allclose(h_new[~m], h_old[~m]):
                    raise RuntimeError("Stage5_3 gate invariant failed: mask_update=false hidden changed.")
        return offsets_g, h_new, eff

    def _compute_record_support_error_all_branches_once_routed(
        self,
        *,
        batch: Dict[str, Any],
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        record_targets: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = source_frame_idx
        _ = route
        targets = list(record_targets) if record_targets is not None else list(batch.get("targets", []))
        if len(targets) == 0:
            raise ValueError("record pass requires non-empty batch['targets'].")
        gaussians_bg_distant, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        num_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0

        support_bg_acc = torch.zeros((num_bg, 1), device=self.device)
        support_distant_acc = torch.zeros((num_distant, 1), device=self.device)
        support_rigid_acc = torch.zeros((num_rigid, 1), device=self.device)
        error_bg_num = torch.zeros((num_bg, 1), device=self.device)
        error_distant_num = torch.zeros((num_distant, 1), device=self.device)
        error_rigid_num = torch.zeros((num_rigid, 1), device=self.device)

        for tgt in targets:
            frame_idx = int(tgt["frame_idx"])
            view = tgt["view"]
            gt_image = tgt["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            gt_image = gt_image.to(self.device)
            height = int(gt_image.shape[0])
            width = int(gt_image.shape[1])

            rigid_idx = torch.zeros((0,), dtype=torch.long, device=self.device)
            if node_state_rigid is not None:
                rigid_mask = self._rigid_point_valid_mask(node_state_rigid, frame_idx)
                rigid_idx = torch.nonzero(rigid_mask, as_tuple=False).squeeze(1)

            parts_means = [gaussians_bg_distant["means"]]
            parts_scales = [gaussians_bg_distant["scales"]]
            parts_quats = [gaussians_bg_distant["quats"]]
            parts_opacities = [gaussians_bg_distant["opacities"]]
            parts_colors = [gaussians_bg_distant["colors"]]
            if node_state_rigid is not None and int(rigid_idx.numel()) > 0:
                point_ids_subset = node_state_rigid.point_ids[rigid_idx, 0]
                means_local = node_state_rigid.means[rigid_idx]
                quats_local = node_state_rigid.quats[rigid_idx]
                parts_means.append(
                    self._transform_rigid_to_world(
                        node_state_rigid,
                        means_local,
                        frame_idx,
                        point_ids_subset=point_ids_subset,
                    )
                )
                parts_quats.append(
                    self._transform_rigid_quats_to_world(
                        node_state_rigid,
                        quats_local,
                        frame_idx,
                        point_ids_subset=point_ids_subset,
                    )
                )
                parts_scales.append(torch.exp(node_state_rigid.scales_log[rigid_idx]))
                parts_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[rigid_idx]).squeeze(-1))
                parts_colors.append(
                    torch.cat(
                        [
                            node_state_rigid.sh_dc[rigid_idx, None, :],
                            node_state_rigid.sh_rest[rigid_idx],
                        ],
                        dim=1,
                    )
                )

            gaussians_scene = {
                "means": torch.cat(parts_means, dim=0),
                "scales": torch.cat(parts_scales, dim=0),
                "quats": torch.cat(parts_quats, dim=0),
                "opacities": torch.cat(parts_opacities, dim=0),
                "colors": torch.cat(parts_colors, dim=0),
            }
            pred_rgb_l, _ = self.alpha_t_extractor.render_rgb_only(
                gaussians_scene,
                [view],
                height,
                width,
                return_acc=True,
                return_debug_stats=False,
            )
            residual = torch.abs(pred_rgb_l[0] - gt_image).mean(dim=-1, keepdim=True)
            error_all, acc_w_all = self._backproject_scene_features_multi_camera(
                gaussians_scene=gaussians_scene,
                source_views=[view],
                features_2d=residual.unsqueeze(0),
                source_pair_valid_mask=torch.ones((1, height, width), dtype=torch.float32, device=self.device),
                height=height,
                width=width,
                backprojector_override=self.stage5_2_record_backprojector,
            )
            num_all = int(gaussians_scene["means"].shape[0])
            if error_all is None:
                error_all = torch.zeros((num_all, 1), device=self.device)
            if acc_w_all is None:
                acc_w_all = torch.zeros((num_all,), device=self.device)

            acc_bg = acc_w_all[:num_bg].unsqueeze(-1)
            err_bg = error_all[:num_bg]
            support_bg_acc = support_bg_acc + acc_bg
            error_bg_num = error_bg_num + err_bg * acc_bg

            if num_distant > 0:
                acc_distant = acc_w_all[num_bg : num_bg + num_distant].unsqueeze(-1)
                err_distant = error_all[num_bg : num_bg + num_distant]
                support_distant_acc = support_distant_acc + acc_distant
                error_distant_num = error_distant_num + err_distant * acc_distant

            if num_rigid > 0 and int(rigid_idx.numel()) > 0:
                acc_rigid = acc_w_all[num_bg + num_distant :].unsqueeze(-1)
                err_rigid = error_all[num_bg + num_distant :]
                support_rigid_acc[rigid_idx] = support_rigid_acc[rigid_idx] + acc_rigid
                error_rigid_num[rigid_idx] = error_rigid_num[rigid_idx] + err_rigid * acc_rigid

        error_bg = torch.where(
            support_bg_acc > 0,
            error_bg_num / (support_bg_acc + float(self.stage5_2_error_eps)),
            torch.zeros_like(error_bg_num),
        )
        error_distant = torch.where(
            support_distant_acc > 0,
            error_distant_num / (support_distant_acc + float(self.stage5_2_error_eps)),
            torch.zeros_like(error_distant_num),
        )
        error_rigid_full = torch.where(
            support_rigid_acc > 0,
            error_rigid_num / (support_rigid_acc + float(self.stage5_2_error_eps)),
            torch.zeros_like(error_rigid_num),
        )
        support_bg = torch.log1p(support_bg_acc)
        support_distant = torch.log1p(support_distant_acc)
        support_rigid_full = torch.log1p(support_rigid_acc)
        return {
            "support_bg": support_bg,
            "error_bg": error_bg,
            "support_distant": support_distant,
            "error_distant": error_distant,
            "support_rigid": support_rigid_full,
            "error_rigid": error_rigid_full,
        }

    def _build_record_targets(self, batch: Dict[str, Any]) -> list[Dict[str, Any]]:
        mode = str(self.stage5_2_record_views)
        if mode != "source_image_refs":
            raise ValueError("Stage5_3 requires record_views=source_image_refs.")
        source_views = list(batch.get("source_views", []))
        source_images = list(batch.get("source_images", []))
        if len(source_views) == 0 or len(source_images) == 0:
            raise ValueError(
                "Stage5_3 record_views=source_image_refs requires non-empty "
                "batch['source_views'] and batch['source_images']."
            )
        if len(source_views) != len(source_images):
            raise ValueError(
                "Stage5_3 source record mismatch: len(source_views) != len(source_images)."
            )
        request_meta = batch.get("request_meta") or {}
        source_refs = request_meta.get("source_image_refs") or []
        source_frame_idx_any = batch.get("source_frame_idx")
        if source_frame_idx_any is None:
            if len(source_refs) > 0:
                source_frame_idx_any = int(source_refs[0][0])
            else:
                targets = list(batch.get("targets", []))
                source_frame_idx_any = int(targets[0]["frame_idx"]) if len(targets) > 0 else 0
        default_source_frame_idx = int(source_frame_idx_any)
        out: list[Dict[str, Any]] = []
        for i, (view, image) in enumerate(zip(source_views, source_images)):
            gt_image = image
            if torch.is_tensor(gt_image) and gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            if i < len(source_refs):
                frame_idx_i = int(source_refs[i][0])
            else:
                frame_idx_i = int(default_source_frame_idx)
            out.append(
                {
                    "frame_idx": int(frame_idx_i),
                    "view": view,
                    "gt_image": gt_image,
                }
            )
        return out

    def _apply_residual_history_update(
        self,
        *,
        history: Dict[str, torch.Tensor],
        error_cur: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> None:
        eb = float(self.stage5_2_error_beta)
        vis = visible_mask.unsqueeze(-1).to(dtype=history["error_ema"].dtype)
        history["error_ema"] = torch.where(
            vis > 0,
            eb * history["error_ema"] + (1.0 - eb) * error_cur.to(dtype=history["error_ema"].dtype),
            history["error_ema"],
        )
        history["initialized"] = torch.maximum(history["initialized"], vis.to(dtype=history["initialized"].dtype))

    def _apply_step_update_norm_ema(
        self,
        *,
        history: Dict[str, torch.Tensor],
        update_norm_cur: torch.Tensor,
    ) -> None:
        ub = float(self.stage5_2_update_norm_beta)
        update_norm_cur = update_norm_cur.to(dtype=history["update_norm_ema"].dtype)
        written_mask = (update_norm_cur.squeeze(-1) > 0).unsqueeze(-1).to(dtype=history["update_norm_ema"].dtype)
        history["update_norm_ema"] = torch.where(
            written_mask > 0,
            ub * history["update_norm_ema"] + (1.0 - ub) * update_norm_cur,
            history["update_norm_ema"],
        )

    def _apply_step_update_norm_ema_from_out(self, out: Dict[str, Any]) -> None:
        key_any = out.get("_cache_key")
        if key_any is None:
            return
        key = (int(key_any[0]), int(key_any[1]))

        node_state_bg = out.get("_node_state_bg")
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")

        num_bg = int(node_state_bg.means.shape[0]) if node_state_bg is not None else 0
        num_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0

        hist_bg = self._get_or_init_history(self.stage5_2_history_bg, key, num_bg)
        hist_distant = self._get_or_init_history(self.stage5_2_history_distant, key, num_distant)
        hist_rigid = self._get_or_init_history(self.stage5_2_history_rigid, key, num_rigid)

        upd_bg = torch.zeros((num_bg, 1), dtype=torch.float32, device=self.device)
        upd_distant = torch.zeros((num_distant, 1), dtype=torch.float32, device=self.device)
        upd_rigid = torch.zeros((num_rigid, 1), dtype=torch.float32, device=self.device)

        idx_bg = out.get("_bg_writeback_idx")
        render_bg = out.get("render_params")
        if (
            node_state_bg is not None
            and render_bg is not None
            and idx_bg is not None
            and int(idx_bg.numel()) > 0
        ):
            delta_bg = render_bg["means_r"][idx_bg].detach() - node_state_bg.means[idx_bg].detach()
            upd_bg[idx_bg] = torch.norm(delta_bg, dim=-1, keepdim=True)

        idx_distant = out.get("_distant_writeback_idx")
        render_distant = out.get("_render_params_distant")
        if (
            node_state_distant is not None
            and render_distant is not None
            and idx_distant is not None
            and int(idx_distant.numel()) > 0
        ):
            delta_d = render_distant["means_r"][idx_distant].detach() - node_state_distant.means[idx_distant].detach()
            upd_distant[idx_distant] = torch.norm(delta_d, dim=-1, keepdim=True)

        idx_rigid = out.get("_rigid_writeback_idx")
        render_rigid_local = out.get("_render_params_rigid_local")
        if (
            node_state_rigid is not None
            and render_rigid_local is not None
            and idx_rigid is not None
            and int(idx_rigid.numel()) > 0
        ):
            if int(render_rigid_local["means_r"].shape[0]) != int(idx_rigid.numel()):
                raise RuntimeError("Stage5_3 update_norm mismatch: rigid writeback rows do not match.")
            delta_r = render_rigid_local["means_r"].detach() - node_state_rigid.means[idx_rigid].detach()
            upd_rigid[idx_rigid] = torch.norm(delta_r, dim=-1, keepdim=True)

        self._apply_step_update_norm_ema(history=hist_bg, update_norm_cur=upd_bg)
        self._apply_step_update_norm_ema(history=hist_distant, update_norm_cur=upd_distant)
        self._apply_step_update_norm_ema(history=hist_rigid, update_norm_cur=upd_rigid)

    def _should_apply_step_update_norm_ema(self) -> bool:
        return bool(self.training) or bool(self.stage5_2_history_update_apply_in_eval)

    def forward(self, batch: Dict) -> Dict[str, Any]:
        out = super().forward(batch)
        if self._should_apply_step_update_norm_ema():
            self._apply_step_update_norm_ema_from_out(out)
        return out

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self.stage5_2_history_bg.clear()
        self.stage5_2_history_distant.clear()
        self.stage5_2_history_rigid.clear()
        self.stage5_2_block_support_bg.clear()
        self.stage5_2_block_support_distant.clear()
        self.stage5_2_block_support_rigid.clear()
        self._stage5_2_last_full_inputs = None

    @torch.no_grad()
    def record_block_history(self, batch: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        key = self._batch_key(batch)
        node_state_bg = self.node_states_bg.get(key)
        if node_state_bg is None:
            raise RuntimeError("Stage5_3 record_block_history: missing node_state_bg.")
        node_state_distant = self.node_states_distant.get(key)
        node_state_rigid = self.node_states_rigid.get(key)
        hist_bg = self._get_or_init_history(self.stage5_2_history_bg, key, int(node_state_bg.means.shape[0]))
        hist_dist = self._get_or_init_history(
            self.stage5_2_history_distant,
            key,
            int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
        )
        hist_rigid = self._get_or_init_history(
            self.stage5_2_history_rigid,
            key,
            int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        )
        self._commit_block_support_to_history(
            key=key,
            history_bg=hist_bg,
            history_distant=hist_dist,
            history_rigid=hist_rigid,
        )
        self._clear_block_support_acc(key)

        record_targets = self._build_record_targets(batch)
        source_frame_idx = int(batch.get("source_frame_idx", record_targets[0]["frame_idx"]))

        N_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        mask_src_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        if node_state_rigid is not None:
            mask_src_rigid = self._rigid_point_valid_mask(node_state_rigid, source_frame_idx)
        S = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
        if node_state_rigid is None:
            route = RigidRoute(
                S=S,
                S_in=S,
                S_out=S,
                inside_mask_S=torch.zeros((0,), dtype=torch.bool, device=self.device),
                route_inside_global=torch.zeros((N_rigid,), dtype=torch.bool, device=self.device),
                means_world_S=torch.zeros((0, 3), device=self.device),
                quats_world_S=torch.zeros((0, 4), device=self.device),
            )
        else:
            route = self._route_rigid_source_points(node_state_rigid, source_frame_idx, S)

        rec = self._compute_record_support_error_all_branches_once_routed(
            batch=batch,
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            record_targets=record_targets,
        )
        self._apply_residual_history_update(
            history=hist_bg,
            error_cur=rec["error_bg"],
            visible_mask=(rec["support_bg"].squeeze(-1) > self.stage5_2_bg_visible_min),
        )
        self._apply_residual_history_update(
            history=hist_dist,
            error_cur=rec["error_distant"],
            visible_mask=(rec["support_distant"].squeeze(-1) > self.stage5_2_distant_visible_min),
        )
        self._apply_residual_history_update(
            history=hist_rigid,
            error_cur=rec["error_rigid"],
            visible_mask=(rec["support_rigid"].squeeze(-1) > self.stage5_2_rigid_visible_min),
        )
        request_meta = batch.get("request_meta") or {}
        source_refs = request_meta.get("source_image_refs")
        if source_refs is None:
            num_source_refs = 1.0 if request_meta.get("source_image_ref") is not None else 0.0
        else:
            num_source_refs = float(len(source_refs))
        return {
            "stage5_2_record_pass_count": 1.0,
            "stage5_2_record_num_views": float(len(record_targets)),
            "stage5_2_record_num_target_refs": float(len(request_meta.get("target_image_refs") or [])),
            "stage5_2_record_num_source_refs": float(num_source_refs),
            "stage5_2_record_use_source_views": 1.0 if str(self.stage5_2_record_views) == "source_image_refs" else 0.0,
            "stage5_2_history_bg_support_mean": float(hist_bg["support_ema"].mean().item()) if hist_bg["support_ema"].numel() > 0 else 0.0,
            "stage5_2_history_bg_error_mean": float(hist_bg["error_ema"].mean().item()) if hist_bg["error_ema"].numel() > 0 else 0.0,
            "stage5_2_history_distant_support_mean": float(hist_dist["support_ema"].mean().item()) if hist_dist["support_ema"].numel() > 0 else 0.0,
            "stage5_2_history_distant_error_mean": float(hist_dist["error_ema"].mean().item()) if hist_dist["error_ema"].numel() > 0 else 0.0,
            "stage5_2_history_rigid_support_mean": float(hist_rigid["support_ema"].mean().item()) if hist_rigid["support_ema"].numel() > 0 else 0.0,
            "stage5_2_history_rigid_error_mean": float(hist_rigid["error_ema"].mean().item()) if hist_rigid["error_ema"].numel() > 0 else 0.0,
            **({"stage5_2_block_exit_block_idx_global": float(event.get("block_idx_global", -1))} if isinstance(event, dict) else {}),
        }


__all__ = ["FullRoutedGRUInputs", "MinimalStreetForwardStage5_3"]
