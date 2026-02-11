from __future__ import annotations

import logging
import os
from pathlib import Path
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from models.feature_extractors import (
    AlphaTWeightExtractor,
    FeatureBackprojector,
    FeatureFusion,
    ImageFeatureExtractor,
)
from models.streetforward.math_utils import _num_sh_bases, get_viewmat
from models.streetforward.node_states import (
    NodeState,
    NodeStateBackground,
    NodeStateRigid,
    NodeStateDistant,
)
from models.streetforward import metrics
from models.streetforward.node_state_mixin import NodeStateMixin, RigidMasks
from models.streetforward.feature_volume_mixin import FeatureVolumeMixin
from models.streetforward.offsets_mixin import OffsetsMixin
from models.streetforward.proxy_rendering_mixin import ProxyRenderingMixin
from models.streetforward.checkpoint_mixin import CheckpointMixin

from gsplat.rendering import rasterization as _gsplat_rasterization
from models.evol_splat import (
    SparseCostRegNet as _SparseCostRegNet,
    construct_sparse_tensor as _construct_sparse_tensor,
    sparse_to_dense_volume as _sparse_to_dense_volume,
)

logger = logging.getLogger(__name__)



class StreetForwardTrainer(CheckpointMixin, ProxyRenderingMixin, OffsetsMixin, FeatureVolumeMixin, NodeStateMixin, nn.Module):
    def __init__(
        self,
        config: OmegaConf,
        device: torch.device = torch.device("cpu"),
        renderer: Optional[Callable] = None,
        sparse_conv: Optional[nn.Module] = None,
        construct_sparse_tensor_fn: Optional[Callable] = None,
        sparse_to_dense_volume_fn: Optional[Callable] = None,
    ):
        """
        初始化 StreetForwardTrainer。
        
        Args:
            config: OmegaConf 配置对象，包含模型、优化器和训练配置
            device: 计算设备（CPU 或 CUDA）
            renderer: 可选的渲染器函数（默认使用 gsplat.rendering.rasterization）
            sparse_conv: 可选的稀疏卷积网络（默认使用 SparseCostRegNet）
            construct_sparse_tensor_fn: 可选的稀疏张量构建函数
            sparse_to_dense_volume_fn: 可选的稀疏到密集体积转换函数
        """
        super().__init__()
        self.config = config
        self.device = device

        # 从配置中读取模型参数
        model_cfg = config.model
        # 偏移量的物理上限（用于 tanh 限制）
        self.offset_max = model_cfg.get("offset_max", 0.1)  # 位置偏移上限（米）
        self.scale_max = model_cfg.get("scale_max", 0.1)  # 尺度偏移上限（对数域）
        self.omega_max = model_cfg.get("omega_max", 0.1)  # 旋转偏移上限（弧度，约5.7°）
        self.opacity_max = model_cfg.get("opacity_max", 0.1)  # 不透明度偏移上限（logit域）
        self.sh_dc_max = model_cfg.get("sh_dc_max", 0.1)  # SH DC偏移上限
        self.sh_rest_max = model_cfg.get("sh_rest_max", 0.05)  # SH rest偏移上限（通常更小）
        # 步长因子（控制偏移量幅度）
        self.eta_means = model_cfg.get("eta_means", 1.0)  # 位置步长因子
        self.eta_scales = model_cfg.get("eta_scales", 1.0)  # 尺度步长因子
        self.eta_opacity = model_cfg.get("eta_opacity", 1.0)  # 不透明度步长因子
        self.eta_sh_dc = model_cfg.get("eta_sh_dc", 1.0)  # SH DC步长因子
        self.eta_sh_rest = model_cfg.get("eta_sh_rest", 1.0)  # SH rest步长因子
        eta_by_node_cfg = model_cfg.get("eta_by_node", None)
        if eta_by_node_cfg is not None:
            try:
                eta_by_node_cfg = OmegaConf.to_container(eta_by_node_cfg, resolve=True)
            except Exception:
                # OmegaConf may not be present or conversion might fail; keep raw config
                pass
        self.eta_by_node = eta_by_node_cfg or {}
        # 其他模型参数
        self.sh_degree = model_cfg.get("sh_degree", 1)  # 球谐函数度数
        self.voxel_size = model_cfg.get("voxel_size", 0.1)  # 体素大小（米）
        self.inner_iterations = model_cfg.get("max_iterations", 1)  # 内部迭代次数
        self.use_2d_features = bool(model_cfg.get("use_2d_features", False))
        self.feat_2d_channels = int(model_cfg.get("feat_2d_channels", 16))
        self.feat_2d_downscale = int(model_cfg.get("feat_2d_downscale", 1))

        bbx_min = model_cfg.get("bbx_min", [-20.0, -20.0, -20.0])
        bbx_max = model_cfg.get("bbx_max", [20.0, 4.8, 70.0])
        self.bbx_min = torch.tensor(bbx_min, dtype=torch.float32, device=device)
        self.bbx_max = torch.tensor(bbx_max, dtype=torch.float32, device=device)

        input_aabb_cfg = model_cfg.get("input_aabb", None)
        if input_aabb_cfg is None and hasattr(config, "data") and hasattr(config.data, "pointcloud"):
            pc_cfg = config.data.pointcloud
            if hasattr(pc_cfg, "get"):
                input_aabb_cfg = pc_cfg.get("input_aabb")
            elif hasattr(pc_cfg, "input_aabb"):
                input_aabb_cfg = pc_cfg.input_aabb
        if input_aabb_cfg is None and hasattr(config, "dataset") and hasattr(config.dataset, "pointcloud"):
            pc_cfg = config.dataset.pointcloud
            if hasattr(pc_cfg, "get"):
                input_aabb_cfg = pc_cfg.get("input_aabb")
            elif hasattr(pc_cfg, "input_aabb"):
                input_aabb_cfg = pc_cfg.input_aabb
        if input_aabb_cfg is None:
            input_aabb_cfg = [bbx_min, bbx_max]
        self.input_aabb_min = torch.tensor(input_aabb_cfg[0], dtype=torch.float32, device=device)
        self.input_aabb_max = torch.tensor(input_aabb_cfg[1], dtype=torch.float32, device=device)

        # Renderer and sparse conv dependencies
        self.renderer = renderer or _gsplat_rasterization
        if self.renderer is None:
            raise ImportError("Renderer not available. Install gsplat or provide a custom renderer.")

        outdim = model_cfg.get("sparseConv_outdim", 32)
        self.feat_3d_dim = outdim
        if sparse_conv is not None:
            self.sparse_conv = sparse_conv.to(device)
        elif _SparseCostRegNet is not None:
            self.sparse_conv = _SparseCostRegNet(d_in=3, d_out=outdim).to(device)
        else:
            raise ImportError("SparseCostRegNet not available. Install models.evol_splat or provide a custom sparse_conv.")

        self.construct_sparse_tensor = construct_sparse_tensor_fn or _construct_sparse_tensor
        if self.construct_sparse_tensor is None:
            raise ImportError("construct_sparse_tensor not available. Install models.evol_splat or provide a custom construct_sparse_tensor_fn.")
        
        self.sparse_to_dense_volume = sparse_to_dense_volume_fn or _sparse_to_dense_volume
        if self.sparse_to_dense_volume is None:
            raise ImportError("sparse_to_dense_volume not available. Install models.evol_splat or provide a custom sparse_to_dense_volume_fn.")

        fused_in_dim = outdim
        if self.use_2d_features:
            # Deep fusion: 6 channels (original RGB + rendered RGB)
            self.image_feature_extractor = ImageFeatureExtractor(
                in_channels=6,  # 3 (original) + 3 (rendered)
                feat_channels=self.feat_2d_channels,
                feature_downscale=self.feat_2d_downscale,
            ).to(device)
            self.alpha_t_extractor = AlphaTWeightExtractor(
                renderer=self.renderer,
                sh_degree=self.sh_degree,
                tile_size=16,
            )
            self.feature_backprojector = FeatureBackprojector()
            # 不再使用可见性通道，保持 3D+2D 融合
            self.feature_fusion = FeatureFusion(use_visibility=False)
            fused_in_dim = outdim + self.feat_2d_channels
        else:
            self.image_feature_extractor = None
            self.alpha_t_extractor = None
            self.feature_backprojector = None
            self.feature_fusion = None

        # GRU-style hidden fusion settings
        self.param_embed_input_dim = 17  # means(3) + rot6d(6) + scales(3) + opacity(1) + sh_dc(3) + sh_rest_energy(1)
        self.param_embed_dim = int(model_cfg.get("param_embed_dim", fused_in_dim))
        self.offset_gru_hidden_dim = int(model_cfg.get("offset_gru_hidden_dim", fused_in_dim))
        self.offset_gru_use_reset_gate = bool(model_cfg.get("offset_gru_use_reset_gate", True))

        # Parameter embedding MLP (shared by bg/rigid/distant)
        self.mlp_params_embed = nn.Sequential(
            nn.Linear(self.param_embed_input_dim, self.param_embed_dim),
            nn.ReLU(),
            nn.Linear(self.param_embed_dim, self.param_embed_dim),
        ).to(device)
        self.param_embed_norm = nn.LayerNorm(self.param_embed_dim).to(device)

        # GRU-style fusion layers
        gru_in_dim = fused_in_dim + self.param_embed_dim
        self.gru_update = nn.Linear(gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim).to(device)
        self.gru_candidate = nn.Linear(gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim).to(device)
        self.gru_reset = (
            nn.Linear(gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim).to(device)
            if self.offset_gru_use_reset_gate
            else None
        )

        # Hidden → offset head projection (keeps head shapes backward compatible)
        if self.offset_gru_hidden_dim != fused_in_dim:
            self.gru_to_head = nn.Linear(self.offset_gru_hidden_dim, fused_in_dim).to(device)
        else:
            self.gru_to_head = nn.Identity()

        # MLP 偏移量预测头（支持 2D+3D 融合特征）
        # 位置偏移预测网络：fused_in_dim → 64 → 32 → 3
        self.mlp_offset_pos = nn.Sequential(
            nn.Linear(fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 输出位置偏移 [N, 3]
        ).to(device)

        # 尺度与旋转偏移预测网络：fused_in_dim → 64 → 32 → 6
        # 输出前3维为尺度偏移，后3维为轴角偏移
        self.mlp_conv = nn.Sequential(
            nn.Linear(fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 3 for scales + 3 for axis-angle
        ).to(device)

        # 不透明度偏移预测网络：fused_in_dim → 64 → 32 → 1
        self.mlp_opacity = nn.Sequential(
            nn.Linear(fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 输出不透明度对数偏移 [N, 1]
        ).to(device)

        # SH 系数偏移预测网络：fused_in_dim → 64 → 32 → 3*num_sh
        num_sh = _num_sh_bases(self.sh_degree)
        self.gaussion_decoder = nn.Sequential(
            nn.Linear(fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),  # 输出SH系数偏移 [N, 3*num_sh]
        ).to(device)

        params: List[torch.nn.Parameter] = []
        params += list(self.sparse_conv.parameters())
        params += list(self.mlp_params_embed.parameters())
        params += list(self.param_embed_norm.parameters())
        params += list(self.gru_update.parameters())
        params += list(self.gru_candidate.parameters())
        if self.gru_reset is not None:
            params += list(self.gru_reset.parameters())
        if not isinstance(self.gru_to_head, nn.Identity):
            params += list(self.gru_to_head.parameters())
        params += list(self.mlp_offset_pos.parameters())
        params += list(self.mlp_conv.parameters())
        params += list(self.mlp_opacity.parameters())
        params += list(self.gaussion_decoder.parameters())
        if self.image_feature_extractor is not None:
            params += list(self.image_feature_extractor.parameters())

        optim_cfg = config.optimizer
        self.optimizer = torch.optim.Adam(
            params,
            lr=optim_cfg.get("lr", 1e-3),
            eps=optim_cfg.get("eps", 1e-15),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )

        # Training infrastructure: AMP, grad clip, LR scheduler
        training_cfg_init = getattr(config, "training", None)
        tc_get = training_cfg_init.get if hasattr(training_cfg_init, "get") else lambda k, d=None: d
        self.use_amp = bool(tc_get("use_amp", False)) and torch.cuda.is_available()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        grad_clip_val = tc_get("grad_clip_max_norm", None)
        self.grad_clip_max_norm = float(grad_clip_val) if grad_clip_val is not None and float(grad_clip_val) > 0 else None

        self.scheduler = None
        sched_cfg = tc_get("lr_scheduler")
        if sched_cfg is not None and (isinstance(sched_cfg, dict) or hasattr(sched_cfg, "get")):
            sched_type = sched_cfg.get("type", "none") or "none"
            max_iter = int(tc_get("max_iterations", 10000))
            if sched_type == "cosine":
                T_max = int(sched_cfg.get("T_max", max_iter))
                eta_min = float(sched_cfg.get("eta_min", 1e-6))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max, eta_min=eta_min
                )
                logger.info(f"LR scheduler: CosineAnnealingLR(T_max={T_max}, eta_min={eta_min})")
            elif sched_type == "step":
                step_size = int(sched_cfg.get("step_size", 3000))
                gamma = float(sched_cfg.get("gamma", 0.5))
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
                logger.info(f"LR scheduler: StepLR(step_size={step_size}, gamma={gamma})")
            elif sched_type == "plateau":
                mode = str(sched_cfg.get("mode", "max"))
                factor = float(sched_cfg.get("factor", 0.5))
                patience = int(sched_cfg.get("patience", 5))
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode=mode, factor=factor, patience=patience
                )
                logger.info(f"LR scheduler: ReduceLROnPlateau(mode={mode}, factor={factor}, patience={patience})")

        if self.use_amp:
            logger.info("AMP (mixed precision) enabled")
        if self.grad_clip_max_norm is not None:
            logger.info(f"Gradient clipping enabled: max_norm={self.grad_clip_max_norm}")

        self.global_step = 0
        self.checkpoint_dir: Optional[str] = None
        self.tb_writer = None
        self.tb_log_every = 50
        self.tb_image_every = None

        training_cfg = getattr(config, "training", {})
        self.checkpoint_dir = (
            training_cfg.get("save_checkpoint_dir", None)
            if hasattr(training_cfg, "get")
            else None
        )
        self.strict_proxy_grad = bool(tc_get("strict_proxy_grad", False))
        self.strict_proxy_grad_steps = int(tc_get("strict_proxy_grad_steps", 0) or 0)
        self.detect_anomaly_steps = int(tc_get("detect_anomaly_steps", 0) or 0)
        sentinel_cfg = tc_get("sentinel", {}) if hasattr(training_cfg_init, "get") else {}
        sget = sentinel_cfg.get if hasattr(sentinel_cfg, "get") else (lambda k, d=None: d)
        self.sentinel_enabled = bool(sget("enabled", False))
        self.sentinel_log_every = int(sget("log_every", self.tb_log_every)) if self.tb_log_every else int(sget("log_every", 1) or 1)
        self.sentinel_alert_on_nan = bool(sget("alert_on_nan", False))
        self.sentinel_alert_on_grad_zero = bool(sget("alert_on_grad_zero", False))
        self.proxy_grad_warn_on_none = bool(sget("warn_on_proxy_grad_none", False))
        max_dense = sget("max_dense_elements", sget("max_vol_elements", None))
        try:
            self.sentinel_max_dense_elements = int(max_dense) if max_dense is not None else None
        except (TypeError, ValueError):
            self.sentinel_max_dense_elements = None
        self._strict_proxy_grad_active = False
        self._strict_checks_active = False
        self._last_sentinel_metrics: Dict[str, float] = {}
        self._last_grad_norms_by_module: Dict[str, float] = {}
        self._last_vol_dim = None
        self._last_vol_dim_prod = None
        self._last_dense_elements_est = None
        # allow both legacy top-level flag and training-level override
        self.log_images = bool(config.get("log_images", False))
        if hasattr(training_cfg, "get"):
            self.log_images = training_cfg.get("log_images", self.log_images)
        self._setup_tensorboard(training_cfg)

        self.node_states: Dict[Tuple[int, int], NodeState] = {}
        self.node_states_bg = self.node_states
        self.node_states_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
        self.node_states_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]] = {}
        # Hidden state caches for GRU-style offsets
        self.h_cache_bg: Dict[Tuple[int, int], torch.Tensor] = {}
        self.h_cache_rigid: Dict[Tuple[int, int], torch.Tensor] = {}
        self.h_cache_distant: Dict[Tuple[int, int], torch.Tensor] = {}
        self._h_cache_signatures: Dict[str, Dict[Tuple[int, int], Tuple[int, ...]]] = {}
        self._lpips_model = None

        # Initialize offset heads to output near-zero offsets
        self._init_offset_heads()

    def _init_offset_heads(self) -> None:
        """
        初始化偏移量预测头，使其输出接近零的偏移量。
        
        这确保训练开始时预测的偏移量接近零，避免初始阶段的大幅跳跃。
        所有偏移预测头的最后一层（输出层）被初始化为零权重和零偏置。
        """
        nn.init.zeros_(self.mlp_offset_pos[-1].weight)
        nn.init.zeros_(self.mlp_offset_pos[-1].bias)
        nn.init.zeros_(self.mlp_conv[-1].weight)
        nn.init.zeros_(self.mlp_conv[-1].bias)
        nn.init.zeros_(self.mlp_opacity[-1].weight)
        nn.init.zeros_(self.mlp_opacity[-1].bias)
        nn.init.zeros_(self.gaussion_decoder[-1].weight)
        nn.init.zeros_(self.gaussion_decoder[-1].bias)

    def _update_runtime_flags(self) -> None:
        """
        Update strict modes (proxy grad, anomaly detection) based on the current global_step.
        """
        step = int(getattr(self, "global_step", 0))
        self._strict_proxy_grad_active = bool(self.strict_proxy_grad) or (
            self.strict_proxy_grad_steps > 0 and step < self.strict_proxy_grad_steps
        )
        self._strict_checks_active = bool(
            self._strict_proxy_grad_active or (self.detect_anomaly_steps > 0 and step < self.detect_anomaly_steps)
        )
        use_anomaly = self.detect_anomaly_steps > 0 and step < self.detect_anomaly_steps
        if torch.is_anomaly_enabled() != use_anomaly:
            torch.autograd.set_detect_anomaly(use_anomaly)

    def _reset_sentinel_cache(self) -> None:
        self._last_sentinel_metrics = {}
        self._last_grad_norms_by_module = {}
        self._last_proxy_grad_norms = {}

    def _check_for_nan_inf(self, tensors: Dict[str, Optional[torch.Tensor]]) -> None:
        if not (self._strict_checks_active or self.sentinel_alert_on_nan):
            return
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            if isinstance(tensor, dict):
                for k, v in tensor.items():
                    self._check_for_nan_inf({f"{name}.{k}": v})
                continue
            if not torch.is_tensor(tensor):
                continue
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{name} contains NaN or Inf.")

    def _compute_grad_norms_by_module(self) -> Dict[str, float]:
        modules = {
            "sparse_conv": self.sparse_conv,
            "mlp_offset_pos": self.mlp_offset_pos,
            "mlp_conv": self.mlp_conv,
            "mlp_opacity": self.mlp_opacity,
            "gaussion_decoder": self.gaussion_decoder,
            "gru_update": self.gru_update,
            "gru_candidate": self.gru_candidate,
        }
        if self.gru_reset is not None:
            modules["gru_reset"] = self.gru_reset
        if hasattr(self, "gru_to_head") and not isinstance(self.gru_to_head, nn.Identity):
            modules["gru_to_head"] = self.gru_to_head
        if self.image_feature_extractor is not None:
            modules["image_feature_extractor"] = self.image_feature_extractor
        if hasattr(self, "feature_fusion") and self.feature_fusion is not None:
            modules["feature_fusion"] = self.feature_fusion

        norms: Dict[str, float] = {}
        for name, module in modules.items():
            total_sq = 0.0
            has_grad = False
            for p in module.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.numel() == 0:
                    continue
                has_grad = True
                total_sq += float(torch.sum(g * g).item())
            norms[name] = float(math.sqrt(total_sq)) if has_grad else 0.0
        return norms

    def _compute_total_grad_norm(self, params: List[torch.nn.Parameter]) -> Optional[float]:
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if len(grads) == 0:
            return None
        total_sq = 0.0
        for g in grads:
            total_sq += float(torch.sum(g * g).item())
        return float(math.sqrt(total_sq))

    def _collect_sentinel_metrics(
        self,
        *,
        targets: List[Dict],
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        masks: RigidMasks,
        render_params_bg: Dict[str, torch.Tensor],
        render_params_rigid: Optional[Dict[str, torch.Tensor]],
        render_params_distant: Optional[Dict[str, torch.Tensor]],
        offsets_bg: Dict[str, torch.Tensor],
        offsets_rigid_world: Optional[Dict[str, torch.Tensor]],
        offsets_distant: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if not (self.sentinel_enabled or self._strict_checks_active or self.sentinel_alert_on_nan or self.sentinel_alert_on_grad_zero):
            return

        metrics: Dict[str, float] = {}
        metrics["num_targets"] = float(len(targets))
        metrics["N_bg"] = float(node_state_bg.means.shape[0])
        metrics["N_rigid"] = float(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0.0
        metrics["N_distant"] = float(node_state_distant.means.shape[0]) if node_state_distant is not None else 0.0

        if masks.mask_update_rigid is not None and masks.mask_update_rigid.numel() > 0:
            metrics["mask_update_rigid_mean"] = float(masks.mask_update_rigid.float().mean().item())
        if masks.mask_src_rigid is not None and masks.mask_src_rigid.numel() > 0:
            metrics["mask_src_rigid_mean"] = float(masks.mask_src_rigid.float().mean().item())
        if masks.idx_tgt_rigid:
            lengths = [int(idx.numel()) for idx in masks.idx_tgt_rigid]
            if len(lengths) > 0:
                metrics["idx_tgt_rigid_mean"] = float(sum(lengths) / len(lengths))
                metrics["idx_tgt_rigid_max"] = float(max(lengths))

        if self._last_vol_dim_prod is not None:
            metrics["vol_dim_prod"] = float(self._last_vol_dim_prod)
        if self._last_dense_elements_est is not None:
            metrics["dense_elements_est"] = float(self._last_dense_elements_est)

        def _render_stats(render_params: Optional[Dict[str, torch.Tensor]], prefix: str) -> None:
            if render_params is None:
                return
            opacities = render_params.get("opacities_r")
            if opacities is not None and opacities.numel() > 0:
                metrics[f"{prefix}_opacities_min"] = float(opacities.min().detach())
                metrics[f"{prefix}_opacities_max"] = float(opacities.max().detach())
            quats = render_params.get("quats_r")
            if quats is not None and quats.numel() > 0:
                quat_norm = torch.linalg.norm(quats.detach(), dim=-1)
                metrics[f"{prefix}_quat_norm_dev"] = float(torch.mean((quat_norm - 1.0).abs()))
            means_r = render_params.get("means_r")
            if means_r is not None and means_r.numel() > 0:
                metrics[f"{prefix}_means_min"] = float(means_r.min().detach())
                metrics[f"{prefix}_means_max"] = float(means_r.max().detach())
            scales_log = render_params.get("scales_log_r")
            if scales_log is not None and scales_log.numel() > 0:
                metrics[f"{prefix}_scales_log_min"] = float(scales_log.min().detach())
                metrics[f"{prefix}_scales_log_max"] = float(scales_log.max().detach())

        def _offset_stats(offsets: Optional[Dict[str, torch.Tensor]], prefix: str) -> None:
            if offsets is None:
                return
            metrics[f"{prefix}_offset_pos_max"] = float(offsets["offset_pos"].detach().abs().max())
            metrics[f"{prefix}_offset_scales_max"] = float(offsets["offset_scales"].detach().abs().max())
            metrics[f"{prefix}_offset_opacity_max"] = float(offsets["offset_opacity"].detach().abs().max())

        _render_stats(render_params_bg, "bg")
        _render_stats(render_params_rigid, "rigid")
        _render_stats(render_params_distant, "distant")

        _offset_stats(offsets_bg, "bg")
        _offset_stats(offsets_rigid_world, "rigid_world")
        _offset_stats(offsets_distant, "distant")

        self._last_sentinel_metrics = metrics

    def _augment_sentinel_with_grads(self, total_loss_val: float) -> None:
        if not (self.sentinel_enabled or self._strict_checks_active or self.sentinel_alert_on_nan or self.sentinel_alert_on_grad_zero):
            return
        metrics = dict(getattr(self, "_last_sentinel_metrics", {}))
        metrics["total_loss"] = float(total_loss_val)
        if self._last_grad_norm is not None:
            metrics["grad_norm_total"] = float(self._last_grad_norm)
        for name, val in self._last_grad_norms_by_module.items():
            metrics[f"grad_{name}"] = float(val)
        for name, val in getattr(self, "_last_proxy_grad_norms", {}).items():
            metrics[f"proxy_grad_{name}"] = float(val)
        if torch.cuda.is_available():
            metrics["max_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
        self._last_sentinel_metrics = metrics

    def _maybe_alert_on_sentinel(self) -> None:
        if not (self.sentinel_enabled or self._strict_checks_active or self.sentinel_alert_on_nan or self.sentinel_alert_on_grad_zero):
            return
        metrics = getattr(self, "_last_sentinel_metrics", {})
        if self._strict_checks_active or self.sentinel_alert_on_nan:
            for name, val in metrics.items():
                if isinstance(val, float) and not math.isfinite(val):
                    raise RuntimeError(f"Sentinel metric {name} is non-finite ({val}).")
        if self.sentinel_alert_on_grad_zero:
            zero_keys = [
                name for name, val in metrics.items()
                if (name.startswith("grad_") or name.startswith("proxy_grad_")) and val == 0.0
            ]
            if zero_keys:
                logger.warning(f"Gradients are zero for: {', '.join(zero_keys)}")
    def _cache_signature(self, node_state: NodeState) -> Tuple[int, ...]:
        """
        Lightweight signature of a node_state to detect point count/id changes for cache alignment.
        """
        num_points = int(node_state.means.shape[0])
        sig = [num_points]
        if hasattr(node_state, "point_ids") and getattr(node_state, "point_ids") is not None:
            point_ids = node_state.point_ids.reshape(-1).to(torch.int64)
            sig.append(int(point_ids.numel()))
            sig.append(int(point_ids.sum().item()))
        return tuple(sig)

    def _get_or_init_hidden(
        self,
        cache: Dict[Tuple[int, int], torch.Tensor],
        key: Tuple[int, int],
        num_points: int,
        node_state: Optional[NodeState] = None,
        node_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Fetch hidden state from cache or initialize zeros. Reset cache when size/signature mismatches.
        """
        h = cache.get(key)
        desired_sig = self._cache_signature(node_state) if node_state is not None else None
        prev_sig = None
        if node_type is not None and hasattr(self, "_h_cache_signatures"):
            prev_sig = self._h_cache_signatures.get(node_type, {}).get(key)

        reset_needed = (
            h is None
            or h.shape[0] != num_points
            or (prev_sig is not None and desired_sig is not None and prev_sig != desired_sig)
        )
        if reset_needed:
            h = torch.zeros(num_points, self.offset_gru_hidden_dim, device=self.device)
            cache[key] = h

        if node_type is not None and desired_sig is not None:
            self._h_cache_signatures.setdefault(node_type, {})[key] = desired_sig
        return h.detach()

    def _setup_tensorboard(self, training_cfg) -> None:
        """
        根据配置初始化 TensorBoard writer。
        
        TensorBoard 是可选的，以保持单元测试轻量级，并避免在非训练上下文中意外写入磁盘。
        
        Args:
            training_cfg: 训练配置字典，需包含 "tensorboard" 键
        """
        tb_cfg = training_cfg.get("tensorboard", {}) if hasattr(training_cfg, "get") else {}
        enabled = tb_cfg.get("enabled", False)
        if not enabled:
            return

        log_dir = tb_cfg.get("log_dir")
        if log_dir is None:
            # Default to run's log_dir/tb when available, otherwise local folder.
            base_log_dir = getattr(self.config, "log_dir", None)
            if base_log_dir is not None:
                log_dir = os.path.join(base_log_dir, "tb")
            else:
                log_dir = "./logs/tensorboard"

        self.tb_log_every = int(tb_cfg.get("log_every", 50))
        self.tb_image_every = tb_cfg.get("log_image_every", None)
        if self.tb_image_every is not None:
            self.tb_image_every = int(self.tb_image_every)

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard not available; install tensorboard to enable logging.")
            return

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=int(tb_cfg.get("flush_secs", 30)))
        logger.info(f"TensorBoard logging enabled at {log_dir}")

    def train_iter(
        self,
        batch: Dict,
        apply_update: bool = True,
        update_state: bool = True,
        evaluate_test: bool = False,
    ) -> Dict:
        """
        执行一次训练迭代。
        
        这是训练流程的主函数，详细流程请参考 docs/trainers/StreetForward_Flow.md。
        
        Args:
            batch: 批次数据字典，需包含：
                - "scene_id": 场景ID
                - "segment_id": 片段ID
                - "source_frame_idx": Source 帧的 frame ID
                - "pointcloud": 点云数据
                - "dynamic_info": 动态物体信息（可选）
                - "targets": Target 帧列表（推荐格式），或
                - "target_views" + "gt_images": 兼容旧格式
                - "test_views" + "test_images": 测试视图（可选，用于评估）
            apply_update: 是否应用优化器更新
            update_state: 是否更新 NodeState
            evaluate_test: 是否评估测试视图
            
        Returns:
            结果字典，包含：
                - "total_loss": 总损失值（标量）
                - "node_state": 更新后的 NodeStateBackground
                - "node_state_rigid": 更新后的 NodeStateRigid（如果存在）
                - "node_state_distant": 更新后的 NodeStateDistant（如果存在）
                - "outputs": 输出列表（如果 log_images=True，包含渲染图像）
                - "test_metrics": 测试指标（如果进行了评估）
        
        训练流程：
        1. 获取或初始化双 NodeState
        2. 解析 target 帧列表
        3. 开始 inner_iterations 循环：
           a. 构建 3D 特征体积（合并静态和动态点云）
           b. 预测偏移量（静态和动态共同预测）
           c. 计算渲染参数（分别应用到两个 NodeState）
           d. 创建代理参数（分别创建静态和动态代理）
           e. 遍历所有 target 帧：
              - 设置 RigidNodes.cur_frame = target.frame_idx
              - 变换动态物体到 target 帧的世界坐标
              - 合并静态和动态参数
              - 渲染图像并计算损失
              - 反向传播到代理参数（梯度累积）
           f. 反向传播到渲染参数（分别处理静态和动态）
           g. 优化器更新（如果 apply_update=True）
           h. 更新双 NodeState（如果 update_state=True）
        4. 保存 NodeState 并返回结果
        """
        key, node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states(batch)
        targets = self._parse_targets(batch)
        self._update_runtime_flags()
        self._reset_sentinel_cache()

        if len(targets) == 0:
            scene_id, segment_id = key
            raise ValueError(
                f"targets is empty: cannot compute loss (scene_id={scene_id}, segment_id={segment_id}). "
                "Check batch construction or _parse_targets."
            )

        outputs: List[Dict] = []
        total_loss_val = 0.0
        test_metrics = None

        self.optimizer.zero_grad(set_to_none=True)

        # Hidden state seeds (detach to stop grad across train_iter)
        h_bg = self._get_or_init_hidden(
            self.h_cache_bg, key, node_state_bg.means.shape[0], node_state=node_state_bg, node_type="bg"
        )
        h_rigid = (
            self._get_or_init_hidden(
                self.h_cache_rigid,
                key,
                node_state_rigid.means.shape[0],
                node_state=node_state_rigid,
                node_type="rigid",
            )
            if node_state_rigid is not None
            else None
        )
        h_distant = (
            self._get_or_init_hidden(
                self.h_cache_distant,
                key,
                node_state_distant.means.shape[0],
                node_state=node_state_distant,
                node_type="distant",
            )
            if node_state_distant is not None
            else None
        )

        for _ in range(self.inner_iterations):
            with torch.cuda.amp.autocast(enabled=getattr(self, "use_amp", False)):
                result = self._train_inner_iteration(
                    batch=batch,
                    targets=targets,
                    node_state_bg=node_state_bg,
                    node_state_rigid=node_state_rigid,
                    node_state_distant=node_state_distant,
                    h_old_bg=h_bg,
                    h_old_rigid=h_rigid,
                    h_old_distant=h_distant,
                )
            total_loss_val += result["loss_val"]
            outputs.extend(result["outputs"])

            if apply_update:
                grad_scaler = getattr(self, "grad_scaler", None)
                grad_clip = getattr(self, "grad_clip_max_norm", None)
                params = [p for g in self.optimizer.param_groups for p in g["params"]]
                self._last_grad_norm = None
                if grad_scaler is not None:
                    grad_scaler.unscale_(self.optimizer)
                if self.sentinel_enabled or self._strict_checks_active:
                    self._last_grad_norms_by_module = self._compute_grad_norms_by_module()
                if grad_scaler is not None:
                    if grad_clip is not None and grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                        self._last_grad_norm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                else:
                    if grad_clip is not None and grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                        self._last_grad_norm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    elif self._last_grad_norm is None:
                        maybe_norm = self._compute_total_grad_norm(params)
                        if maybe_norm is not None:
                            self._last_grad_norm = maybe_norm
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

            h_bg = result["h_new_bg"]
            h_rigid = result["h_new_rigid"]
            h_distant = result["h_new_distant"]

            if update_state:
                self._update_node_states(
                    render_params_bg=result["render_params_bg"],
                    render_params_rigid=result["render_params_rigid"],
                    render_params_distant=result["render_params_distant"],
                    node_state_bg=node_state_bg,
                    node_state_rigid=node_state_rigid,
                    node_state_distant=node_state_distant,
                )

        self.node_states[key] = node_state_bg.detach_clone()
        self.node_states_rigid[key] = node_state_rigid.detach_clone() if node_state_rigid is not None else None
        self.node_states_distant[key] = node_state_distant.detach_clone() if node_state_distant is not None else None

        if update_state:
            self.h_cache_bg[key] = h_bg.detach()
            if node_state_rigid is not None:
                self.h_cache_rigid[key] = h_rigid.detach()
            if node_state_distant is not None:
                self.h_cache_distant[key] = h_distant.detach()

        if self._strict_checks_active or self.sentinel_alert_on_nan:
            if not math.isfinite(total_loss_val):
                raise RuntimeError("Total loss is NaN or Inf.")
        self._augment_sentinel_with_grads(total_loss_val)
        self._maybe_alert_on_sentinel()

        if apply_update:
            self.global_step += 1
            self._log_to_tensorboard(total_loss_val, outputs)

        if evaluate_test and batch.get("test_views"):
            prev_mode = self.training
            self.eval()
            with torch.no_grad():
                test_metrics = self._evaluate_test_views(
                    node_state=self.node_states[key],
                    test_views=batch.get("test_views", []),
                    test_images=batch.get("test_images", []),
                )
            if prev_mode:
                self.train()

        return {
            "total_loss": torch.tensor(total_loss_val, device=self.device),
            "node_state": self.node_states[key],
            "node_state_rigid": self.node_states_rigid.get(key),
            "node_state_distant": self.node_states_distant.get(key),
            "outputs": outputs,
            "test_metrics": test_metrics,
        }

    def _parse_targets(self, batch: Dict) -> List[Dict]:
        """
        统一解析 target 视角，兼容旧字段。
        """
        targets: List[Dict] = []
        if "targets" in batch:
            for target in batch["targets"]:
                targets.append(
                    {
                        "frame_idx": target.get("frame_idx", batch.get("source_frame_idx", 0)),
                        "view": target["view"],
                        "gt_image": target["gt_image"],
                    }
                )
        else:
            target_views = batch.get("target_views", [])
            gt_images = batch.get("gt_images", [])
            for view, gt_img in zip(target_views, gt_images):
                targets.append(
                    {
                        "frame_idx": batch.get("source_frame_idx", 0),
                        "view": view,
                        "gt_image": gt_img,
                    }
                )
        return targets

    def _get_source_frame_idx(self, batch: Dict) -> int:
        source_frame_idx = batch.get("source_frame_idx")
        if source_frame_idx is None:
            raise ValueError(
                "source_frame_idx is required but not found in batch. "
                "Please ensure the batch contains source_frame_idx."
            )
        return int(source_frame_idx)

    def _train_inner_iteration(
        self,
        batch: Dict,
        targets: List[Dict],
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        h_old_bg: torch.Tensor,
        h_old_rigid: Optional[torch.Tensor],
        h_old_distant: Optional[torch.Tensor],
    ) -> Dict:
        """
        单次 inner iteration 的编排：构建特征 → 偏移 → 渲染参数 → 渲染 & 反传。
        """
        source_frame_idx = self._get_source_frame_idx(batch)
        masks = self._precompute_rigid_masks(node_state_rigid, source_frame_idx, targets)

        feat_bg, feat_rigid, rigid_visible_mask, _rigid_in_crop_mask = self._build_3d_feature_volume(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            source_frame_idx=source_frame_idx,
            mask_src_rigid=masks.mask_src_rigid,
            idx_src_rigid=masks.idx_src_rigid,
        )

        feat_bg_input, feat_rigid_input, feat_distant_input, feat_2d_bg, feat_2d_rigid, feat_2d_distant = self._compute_and_fuse_features(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            node_state_distant=node_state_distant,
            source_frame_idx=source_frame_idx,
            rigid_visible_mask=rigid_visible_mask,
            feat_bg=feat_bg,
            feat_rigid=feat_rigid,
            source_views=batch.get("source_views", []),
            source_images=batch.get("src_images", []),
        )

        self._last_feat_3d_bg = feat_bg
        self._last_feat_3d_rigid = feat_rigid
        self._last_feat_3d_distant = (
            feat_distant_input[:, : self.feat_3d_dim].detach().clone()
            if feat_distant_input is not None and feat_distant_input.numel() > 0
            else None
        )
        self._last_feat_2d_bg = feat_2d_bg
        self._last_feat_2d_rigid = feat_2d_rigid
        self._last_feat_2d_distant = feat_2d_distant
        self._last_feat_bg_input = feat_bg_input
        self._last_feat_rigid_input = feat_rigid_input
        self._last_feat_distant_input = feat_distant_input

        # Build params for embedding (transform rigid to world for alignment)
        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        params_rigid = None
        if node_state_rigid is not None:
            params_rigid = self._build_params_for_embed(node_state_rigid, coord_space="world", frame_idx=source_frame_idx)
        params_distant = None
        if node_state_distant is not None:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")

        offsets_bg, h_new_bg = self._predict_offsets_gru(
            feat=feat_bg_input,
            params_for_embed=params_bg,
            h_old=h_old_bg,
            mask_update_rigid=None,
        )

        offsets_rigid_world, h_new_rigid = None, h_old_rigid
        if node_state_rigid is not None and feat_rigid_input is not None and feat_rigid_input.numel() > 0:
            offsets_rigid_world, h_new_rigid = self._predict_offsets_gru(
                feat=feat_rigid_input,
                params_for_embed=params_rigid,
                h_old=h_old_rigid if h_old_rigid is not None else torch.zeros(
                    node_state_rigid.means.shape[0],
                    self.offset_gru_hidden_dim,
                    device=self.device,
                ),
                mask_update_rigid=masks.mask_update_rigid,
            )
        elif node_state_rigid is not None:
            # keep hidden unchanged if no features
            h_new_rigid = h_old_rigid

        offsets_distant, h_new_distant = None, h_old_distant
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            offsets_distant, h_new_distant = self._predict_offsets_gru(
                feat=feat_distant_input,
                params_for_embed=params_distant,
                h_old=h_old_distant if h_old_distant is not None else torch.zeros(
                    node_state_distant.means.shape[0],
                    self.offset_gru_hidden_dim,
                    device=self.device,
                ),
                mask_update_rigid=None,
            )

        self._last_offsets_bg = offsets_bg
        self._last_offsets_rigid = offsets_rigid_world
        self._last_offsets_distant = offsets_distant

        if masks.mask_update_rigid is not None and offsets_rigid_world is not None:
            offset_pos_gated = offsets_rigid_world["offset_pos"][~masks.mask_update_rigid]
            if offset_pos_gated.numel() > 0:
                max_gated = offset_pos_gated.abs().max().item()
                if max_gated > 1e-6:
                    import warnings
                    warnings.warn(
                        f"[Sanity Check A] Offsets for points without supervision should be gated to 0, "
                        f"but max abs value is {max_gated:.2e}. This may indicate a bug."
                    )

        render_params_bg, render_params_rigid, render_params_distant = self._compute_render_params_for_inner_iter(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            node_state_distant=node_state_distant,
            offsets_bg=offsets_bg,
            offsets_rigid_world=offsets_rigid_world,
            offsets_distant=offsets_distant,
            source_frame_idx=source_frame_idx,
        )

        self._check_for_nan_inf(
            {
                "offsets_bg": offsets_bg,
                "offsets_rigid": offsets_rigid_world,
                "offsets_distant": offsets_distant,
                "render_params_bg": render_params_bg,
                "render_params_rigid": render_params_rigid,
                "render_params_distant": render_params_distant,
            }
        )

        proxies_bg = self._create_proxy_params(render_params_bg)
        proxies_rigid = self._create_proxy_params(render_params_rigid) if render_params_rigid is not None else None
        proxies_distant = self._create_proxy_params(render_params_distant) if render_params_distant is not None else None

        total_loss, outputs = self._render_targets_and_accumulate_loss(
            targets=targets,
            proxies_bg=proxies_bg,
            proxies_rigid=proxies_rigid,
            proxies_distant=proxies_distant,
            node_state_rigid=node_state_rigid,
            masks=masks,
        )

        self._backward_to_render_params(
            render_params_bg=render_params_bg,
            render_params_rigid=render_params_rigid,
            render_params_distant=render_params_distant,
            proxies_bg=proxies_bg,
            proxies_rigid=proxies_rigid,
            proxies_distant=proxies_distant,
        )

        self._collect_sentinel_metrics(
            targets=targets,
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            node_state_distant=node_state_distant,
            masks=masks,
            render_params_bg=render_params_bg,
            render_params_rigid=render_params_rigid,
            render_params_distant=render_params_distant,
            offsets_bg=offsets_bg,
            offsets_rigid_world=offsets_rigid_world,
            offsets_distant=offsets_distant,
        )

        return {
            "loss_val": total_loss,
            "outputs": outputs,
            "render_params_bg": render_params_bg,
            "render_params_rigid": render_params_rigid,
            "render_params_distant": render_params_distant,
            "h_new_bg": h_new_bg,
            "h_new_rigid": h_new_rigid,
            "h_new_distant": h_new_distant,
        }

    def forward(self, batch: Dict) -> Dict:
        return self.train_iter(batch)

    def _evaluate_test_views(
        self,
        node_state: NodeState,
        test_views: List,
        test_images: List[torch.Tensor],
    ) -> Optional[Dict[str, float]]:
        """
        评估测试视图的性能指标（无梯度更新）。
        
        Args:
            node_state: NodeState（用于渲染）
            test_views: 测试视角列表
            test_images: 测试图像列表
            
        Returns:
            评估指标字典，包含：
                - "psnr": 平均 PSNR（峰值信噪比）
                - "ssim": 平均 SSIM（结构相似性）
                - "lpips": 平均 LPIPS（感知相似性）
                - "num_test_views": 测试视图数量
            如果没有测试视图，返回 None
        """
        if test_views is None or len(test_views) == 0:
            return None

        render_params = self._compute_render_params(node_state)

        def render_fn(view, height, width):
            return self._render_single_view(render_params, view, height, width)

        metrics_result, self._lpips_model = metrics.evaluate_test_views(
            render_fn=render_fn,
            test_views=test_views,
            test_images=test_images,
            device=self.device,
            lpips_model=self._lpips_model,
        )
        return metrics_result

    def evaluate(self, batch: Dict) -> Dict[str, float]:
        """
        评估模型在测试视图上的性能（无梯度更新）。
        
        Args:
            batch: 批次数据字典，需包含 "test_views" 和 "test_images"
            
        Returns:
            评估指标字典，包含 PSNR、SSIM、LPIPS 等
            如果没有测试视图，返回空字典
        """
        self.eval()
        key, node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states(batch)
        # 评估时只需要使用 node_state_bg，因为测试视图通常不包含动态物体
        metrics = self._evaluate_test_views(
            node_state=node_state_bg,
            test_views=batch.get("test_views", []),
            test_images=batch.get("test_images", []),
        )
        self.train()
        return metrics or {}

    def step_scheduler_plateau(self, metric: float) -> None:
        """
        Step ReduceLROnPlateau scheduler (call after eval with validation metric).
        No-op if scheduler is not ReduceLROnPlateau.
        """
        if self.scheduler is not None and isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(metric)
            new_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Plateau scheduler stepped: metric={metric:.4f}, new_lr={new_lr:.2e}")

    def _log_to_tensorboard(self, total_loss_val: float, outputs: List[Dict]) -> None:
        """
        在启用时向 TensorBoard 写入标量和图像。
        
        Args:
            total_loss_val: 总损失值
            outputs: 输出列表（如果 log_images=True，包含渲染图像）
        """
        if self.tb_writer is None:
            return

        step = self.global_step
        if self.tb_log_every and step % self.tb_log_every == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb_writer.add_scalar("train/total_loss", total_loss_val, step)
            self.tb_writer.add_scalar("train/lr", lr, step)
            grad_norm = getattr(self, "_last_grad_norm", None)
            if grad_norm is not None:
                self.tb_writer.add_scalar("train/grad_norm", grad_norm, step)

        if self.sentinel_enabled and self.sentinel_log_every and step % self.sentinel_log_every == 0:
            metrics = getattr(self, "_last_sentinel_metrics", {})
            for name, val in metrics.items():
                if isinstance(val, float):
                    self.tb_writer.add_scalar(f"sentinel/{name}", val, step)

        if (
            self.log_images
            and self.tb_image_every is not None
            and outputs
            and step % self.tb_image_every == 0
        ):
            for idx, out in enumerate(outputs):
                if "rgb" not in out:
                    continue
                rgb = out["rgb"]
                acc = out.get("acc", None)
                tag_prefix = f"train/view_{idx}"
                if rgb.dim() == 3:
                    self.tb_writer.add_image(tag_prefix + "/rgb", rgb.permute(2, 0, 1), step)
                if acc is not None:
                    if acc.dim() == 2:
                        self.tb_writer.add_image(tag_prefix + "/alpha", acc.unsqueeze(0), step)
                    elif acc.dim() == 3:
                        self.tb_writer.add_image(tag_prefix + "/alpha", acc, step)
                # only log the first view to limit disk usage
                break

    def close(self) -> None:
        """
        关闭 TensorBoard writer（如果已创建）。
        
        应在训练结束时调用，确保所有日志都已写入磁盘。
        """
        if self.tb_writer is not None:
            self.tb_writer.close()

__all__ = ["StreetForwardTrainer", "NodeState", "NodeStateBackground", "NodeStateRigid", "NodeStateDistant"]
