from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.feature_extractors import AlphaTWeightExtractorV4
from models.streetforward.minimal_trainer_stage4_0 import merge_debug_stats_as_perf_floats
from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_3 import (
    AttributeGate,
    FullRoutedGRUInputs,
    MinimalStreetForwardStage5_3,
)
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid


class CurrentObsEmbed(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, obs_code: torch.Tensor) -> torch.Tensor:
        if obs_code.numel() == 0:
            return obs_code.new_zeros((0, int(self.net[0].out_features)))
        return self.net(obs_code)


class MinimalStreetForwardStage5_4(MinimalStreetForwardStage5_3):
    def __init__(self, config, device: torch.device, **kwargs):
        self._stage5_4_obs_code_all: Optional[torch.Tensor] = None
        self._stage5_4_active_obs: Optional[Dict[str, Optional[torch.Tensor]]] = None
        super().__init__(config=config, device=device, **kwargs)
        self._init_stage5_4_extractor()

    def _validate_stage5_3_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model"))
        if stage != "5_4":
            raise ValueError("Stage5_4 requires model.stage='5_4'.")

        # Reuse Stage5_3 validations by temporarily remapping stage key.
        old_stage = model_cfg.get("stage")
        model_cfg["stage"] = "5_3"
        try:
            super()._validate_stage5_3_config(config)
        finally:
            model_cfg["stage"] = old_stage

        backprojector_version = str(self._require_key(model_cfg, "backprojector_version", "model")).strip().lower()
        if backprojector_version != "v4":
            raise ValueError("Stage5_4 requires model.backprojector_version='v4'.")

        obs_cfg = self._require_key(config, "current_observation", "config")
        if bool(self._require_key(obs_cfg, "enable", "current_observation")) is not True:
            raise ValueError("Stage5_4 requires current_observation.enable=true.")
        if int(obs_cfg.get("dim", 2)) != 2:
            raise ValueError("Stage5_4 requires current_observation.dim=2.")
        if str(obs_cfg.get("rho_source", "feature")).strip().lower() != "feature":
            raise ValueError("Stage5_4 requires current_observation.rho_source=feature.")
        if bool(obs_cfg.get("record_to_history_memory", False)):
            raise ValueError("Stage5_4 requires current_observation.record_to_history_memory=false.")

    def _init_stage5_3_modules(self, config) -> None:
        super()._init_stage5_3_modules(config)
        obs_cfg = self._require_key(config, "current_observation", "config")
        obs_dim = int(obs_cfg.get("dim", 2))
        if obs_dim != 2:
            raise ValueError(f"Stage5_4 expects current_observation.dim=2, got {obs_dim}.")

        self.stage5_4_obs_enabled = bool(obs_cfg.get("enable", True))
        self.stage5_4_obs_eps = float(obs_cfg.get("eps", 1.0e-6))
        self.stage5_4_input_to_struct_decoder = bool(obs_cfg.get("input_to_struct_decoder", True))
        self.stage5_4_input_to_far_mlp = bool(obs_cfg.get("input_to_far_mlp", True))
        self.stage5_4_input_to_gru = bool(obs_cfg.get("input_to_gru", True))
        self.stage5_4_input_to_history_gate = bool(obs_cfg.get("input_to_history_gate", True))

        feat2d_dim = int(self.stage5_2_feat_2d_channels)
        fused_dim = int(self.fused_in_dim)
        self.current_obs_struct_embed = CurrentObsEmbed(feat2d_dim).to(self.device)
        self.current_obs_far_embed = CurrentObsEmbed(feat2d_dim).to(self.device)
        self.current_obs_gru_embed = CurrentObsEmbed(fused_dim).to(self.device)
        self.current_obs_gate_embed = CurrentObsEmbed(fused_dim).to(self.device)

    def _init_stage5_4_extractor(self) -> None:
        self.alpha_t_extractor_v4 = AlphaTWeightExtractorV4(
            renderer=self.renderer,
            sh_degree=self.sh_degree,
            tile_size=16,
        )
        if not self.alpha_t_extractor_v4.fused_multi_camera_obs_available:
            raise RuntimeError("Stage5_4 fast-fail: AlphaTWeightExtractorV4 fused op is unavailable.")

    def _split_obs_code(
        self,
        *,
        num_bg: int,
        num_distant: int,
        num_rigid_s: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        total = int(num_bg + num_distant + num_rigid_s)
        if total == 0:
            empty = torch.zeros((0, 2), device=device, dtype=dtype)
            return empty, None, None
        obs_all = self._stage5_4_obs_code_all
        if obs_all is None:
            raise RuntimeError("Stage5_4 expected obs_code from V4 backprojection, got None.")
        if obs_all.dim() != 2 or int(obs_all.shape[1]) != 2:
            raise RuntimeError(f"Stage5_4 obs_code must have shape [N,2], got {tuple(obs_all.shape)}.")
        if int(obs_all.shape[0]) != total:
            raise RuntimeError(
                f"Stage5_4 obs_code length mismatch: obs={obs_all.shape[0]}, "
                f"expected bg+distant+rigid_S={total} ({num_bg}+{num_distant}+{num_rigid_s})."
            )
        obs_all = obs_all.to(device=device, dtype=dtype)

        i0 = 0
        i1 = i0 + num_bg
        i2 = i1 + num_distant
        i3 = i2 + num_rigid_s
        obs_bg = obs_all[i0:i1]
        obs_distant = obs_all[i1:i2] if num_distant > 0 else None
        obs_rigid = obs_all[i2:i3] if num_rigid_s > 0 else None
        return obs_bg, obs_distant, obs_rigid

    @staticmethod
    def _apply_obs_feat_add(
        feat_2d: Optional[torch.Tensor],
        obs: Optional[torch.Tensor],
        embed: CurrentObsEmbed,
    ) -> Optional[torch.Tensor]:
        if feat_2d is None or obs is None:
            return feat_2d
        if int(feat_2d.shape[0]) != int(obs.shape[0]):
            return feat_2d
        return feat_2d + embed(obs).to(device=feat_2d.device, dtype=feat_2d.dtype)

    def _backproject_scene_features_multi_camera(
        self,
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        features_2d: torch.Tensor,
        source_pair_valid_mask: torch.Tensor,
        height: int,
        width: int,
        backprojector_override=None,
        return_debug_stats: Optional[bool] = None,
        return_raw_lift: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_gaussians = int(gaussians_scene["means"].shape[0])
        if num_gaussians == 0:
            self._stage5_4_obs_code_all = None
            if bool(return_raw_lift):
                return None, None, None
            return None, None
        backprojector_impl = backprojector_override if backprojector_override is not None else self.feature_backprojector
        debug_stats = True if return_debug_stats is None else bool(return_debug_stats)
        bp_out = self.alpha_t_extractor_v4.render_and_backproject_streaming_fused_multi_camera(
            gaussians=gaussians_scene,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=num_gaussians,
            backprojector=backprojector_impl,
            source_pair_valid_mask=source_pair_valid_mask,
            return_accumulated_weights=True,
            return_obs_code=True,
            return_debug_stats=debug_stats,
            return_raw_sums=bool(return_raw_lift),
        )
        if bool(return_raw_lift):
            if debug_stats:
                feat_sum, weight_sum_feature, acc_w, obs_code, bp_stats = bp_out
            else:
                feat_sum, weight_sum_feature, acc_w, obs_code = bp_out
                bp_stats = {}
        elif debug_stats:
            feat_2d_all, acc_w, obs_code, bp_stats = bp_out
        else:
            feat_2d_all, acc_w, obs_code = bp_out
            bp_stats = {}
        if debug_stats:
            merge_debug_stats_as_perf_floats(self._perf_acc, "2d_bp_scene_", bp_stats)
        self._perf_acc["2d_bp_scene_call_count"] = float(self._perf_acc.get("2d_bp_scene_call_count", 0.0) + 1.0)
        self._stage5_4_obs_code_all = obs_code.detach()
        if bool(return_raw_lift):
            return feat_sum, weight_sum_feature, acc_w
        return feat_2d_all, acc_w

    def _build_struct_decoder_input_near(self, **kwargs) -> Any:
        feat_2d_bg = kwargs.get("feat_2d_bg")
        feat_2d_rigid_s = kwargs.get("feat_2d_rigid_S")
        route: RigidRoute = kwargs["route"]
        obs_state = self._stage5_4_active_obs
        if bool(self.stage5_4_input_to_struct_decoder) and obs_state is not None:
            obs_bg = obs_state.get("obs_bg")
            obs_rigid = obs_state.get("obs_rigid")
            feat_2d_bg = self._apply_obs_feat_add(feat_2d_bg, obs_bg, self.current_obs_struct_embed)
            if feat_2d_rigid_s is not None and obs_rigid is not None and int(obs_rigid.shape[0]) == int(feat_2d_rigid_s.shape[0]):
                feat_2d_rigid_s = self._apply_obs_feat_add(feat_2d_rigid_s, obs_rigid, self.current_obs_struct_embed)
        kwargs["feat_2d_bg"] = feat_2d_bg
        kwargs["feat_2d_rigid_S"] = feat_2d_rigid_s
        return super()._build_struct_decoder_input_near(**kwargs)

    def _build_struct_decoder_input_far(self, **kwargs) -> Any:
        feat_2d_distant = kwargs.get("feat_2d_distant")
        feat_2d_rigid_s = kwargs.get("feat_2d_rigid_S")
        obs_state = self._stage5_4_active_obs
        if bool(self.stage5_4_input_to_far_mlp) and obs_state is not None:
            obs_distant = obs_state.get("obs_distant")
            obs_rigid = obs_state.get("obs_rigid")
            feat_2d_distant = self._apply_obs_feat_add(feat_2d_distant, obs_distant, self.current_obs_far_embed)
            if feat_2d_rigid_s is not None and obs_rigid is not None and int(obs_rigid.shape[0]) == int(feat_2d_rigid_s.shape[0]):
                feat_2d_rigid_s = self._apply_obs_feat_add(feat_2d_rigid_s, obs_rigid, self.current_obs_far_embed)
        kwargs["feat_2d_distant"] = feat_2d_distant
        kwargs["feat_2d_rigid_S"] = feat_2d_rigid_s
        return super()._build_struct_decoder_input_far(**kwargs)

    def _compute_gate(self, *, feat: Optional[torch.Tensor], branch_id: int, **kwargs) -> Optional[AttributeGate]:
        obs_state = self._stage5_4_active_obs
        if bool(self.stage5_4_input_to_history_gate) and obs_state is not None and feat is not None:
            obs_branch = None
            if int(branch_id) == 0:
                obs_branch = obs_state.get("obs_bg")
            elif int(branch_id) == 1:
                obs_branch = obs_state.get("obs_rigid_in")
            elif int(branch_id) == 2:
                obs_branch = obs_state.get("obs_distant")
            elif int(branch_id) == 3:
                obs_branch = obs_state.get("obs_rigid_out")
            if obs_branch is not None and int(obs_branch.shape[0]) == int(feat.shape[0]):
                feat = feat + self.current_obs_gate_embed(obs_branch).to(device=feat.device, dtype=feat.dtype)
        return super()._compute_gate(feat=feat, branch_id=branch_id, **kwargs)

    def _compute_full_routed_gru_inputs(self, **kwargs) -> FullRoutedGRUInputs:
        feat_2d_bg = kwargs["feat_2d_bg"]
        feat_2d_distant = kwargs.get("feat_2d_distant")
        feat_2d_rigid_s = kwargs.get("feat_2d_rigid_S")
        route: RigidRoute = kwargs["route"]
        num_bg = int(feat_2d_bg.shape[0])
        num_distant = int(feat_2d_distant.shape[0]) if feat_2d_distant is not None else 0
        num_rigid_s = int(feat_2d_rigid_s.shape[0]) if feat_2d_rigid_s is not None else 0
        obs_bg, obs_distant, obs_rigid = self._split_obs_code(
            num_bg=num_bg,
            num_distant=num_distant,
            num_rigid_s=num_rigid_s,
            device=feat_2d_bg.device,
            dtype=feat_2d_bg.dtype,
        )
        obs_rigid_in = None
        obs_rigid_out = None
        if obs_rigid is not None and int(obs_rigid.shape[0]) == int(route.inside_mask_S.shape[0]):
            obs_rigid_in = obs_rigid[route.inside_mask_S]
            obs_rigid_out = obs_rigid[~route.inside_mask_S]

        self._stage5_4_active_obs = {
            "obs_bg": obs_bg,
            "obs_distant": obs_distant,
            "obs_rigid": obs_rigid,
            "obs_rigid_in": obs_rigid_in,
            "obs_rigid_out": obs_rigid_out,
        }
        try:
            full = super()._compute_full_routed_gru_inputs(**kwargs)
        finally:
            self._stage5_4_active_obs = None

        if not bool(self.stage5_4_input_to_gru):
            return full

        if full.feat_bg_input is not None and int(full.feat_bg_input.shape[0]) == int(obs_bg.shape[0]):
            full.feat_bg_input = full.feat_bg_input + self.current_obs_gru_embed(obs_bg).to(
                device=full.feat_bg_input.device, dtype=full.feat_bg_input.dtype
            )
        if full.feat_distant_input is not None and obs_distant is not None and int(full.feat_distant_input.shape[0]) == int(obs_distant.shape[0]):
            full.feat_distant_input = full.feat_distant_input + self.current_obs_gru_embed(obs_distant).to(
                device=full.feat_distant_input.device, dtype=full.feat_distant_input.dtype
            )
        if full.feat_rigid_in_input_all is not None and obs_rigid_in is not None and int(full.feat_rigid_in_input_all.shape[0]) == int(obs_rigid_in.shape[0]):
            full.feat_rigid_in_input_all = full.feat_rigid_in_input_all + self.current_obs_gru_embed(obs_rigid_in).to(
                device=full.feat_rigid_in_input_all.device, dtype=full.feat_rigid_in_input_all.dtype
            )
        if full.feat_rigid_out_input_all is not None and obs_rigid_out is not None and int(full.feat_rigid_out_input_all.shape[0]) == int(obs_rigid_out.shape[0]):
            full.feat_rigid_out_input_all = full.feat_rigid_out_input_all + self.current_obs_gru_embed(obs_rigid_out).to(
                device=full.feat_rigid_out_input_all.device, dtype=full.feat_rigid_out_input_all.dtype
            )
        return full


__all__ = ["MinimalStreetForwardStage5_4"]
