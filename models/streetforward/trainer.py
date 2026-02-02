from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
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
from models.streetforward.logging_utils import _debug_log
from models.streetforward.math_utils import _num_sh_bases, get_viewmat
from models.streetforward.node_states import (
    NodeState,
    NodeStateBackground,
    NodeStateRigid,
    NodeStateDistant,
)
from models.streetforward.node_state_mixin import NodeStateMixin
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
        # allow both legacy top-level flag and training-level override
        self.log_images = bool(config.get("log_images", False))
        if hasattr(training_cfg, "get"):
            self.log_images = training_cfg.get("log_images", self.log_images)
        self._setup_tensorboard(training_cfg)

        self.node_states: Dict[Tuple[int, int], NodeState] = {}
        self.node_states_bg = self.node_states
        self.node_states_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
        self.node_states_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]] = {}
        self._lpips_model = None
        self._lpips_unavailable = False
        self._ssim_unavailable = False
        
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
        targets = []
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

        # Skip if no target views (no supervision)
        if len(targets) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device),
                "node_state": node_state_bg,
                "node_state_rigid": node_state_rigid,
                "node_state_distant": node_state_distant,
                "outputs": [],
            }
        
        view_count = len(targets)
        outputs = []
        total_loss_val = 0.0  # Use scalar to avoid keeping computation graph
        test_metrics = None

        self.optimizer.zero_grad(set_to_none=True)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:train_iter",
                "Before inner iterations",
                {
                    "num_targets": len(targets),
                    "inner_iterations": self.inner_iterations,
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                },
                hypothesis_id="H4",
            )
        # #endregion

        for inner_iter_idx in range(self.inner_iterations):
            source_frame_idx = batch.get("source_frame_idx")
            if source_frame_idx is None:
                raise ValueError(
                    "source_frame_idx is required but not found in batch. "
                    "Please ensure the batch contains source_frame_idx."
                )
            source_frame_idx = int(source_frame_idx)
            
            # ===== 新增：预计算 rigid 可见性 mask（方案 1：最小改动） =====
            mask_src_rigid = None
            mask_tgt_rigid = []
            mask_any_tgt_rigid = None
            mask_update_rigid = None
            idx_tgt_rigid = []
            idx_src_rigid = None
            
            if node_state_rigid is not None:
                with torch.no_grad():
                    # Nr = rigid 点数（local coords）
                    Nr = node_state_rigid.means.shape[0]
                    
                    # 1) pose_valid：按 instance 粒度 -> per-point mask
                    pose_valid_src = self._per_point_pose_valid(node_state_rigid, source_frame_idx)  # [Nr] bool
                    pose_valid_tgt = []  # list of [Nr] bool
                    
                    for tgt in targets:
                        pose_valid_tgt.append(self._per_point_pose_valid(node_state_rigid, tgt["frame_idx"]))
                    
                    # 2) visible：使用 instances_fv
                    visible_src = self._visible_mask_from_instances_fv(node_state_rigid, source_frame_idx)  # [Nr] bool
                    
                    visible_tgt = []
                    for tgt in targets:
                        visible_tgt.append(self._visible_mask_from_instances_fv(node_state_rigid, tgt["frame_idx"]))
                    
                    # 3) 组合 mask
                    mask_src_rigid = pose_valid_src & visible_src  # [Nr]
                    mask_tgt_rigid = [pv & vis for pv, vis in zip(pose_valid_tgt, visible_tgt)]  # list([Nr])
                    mask_any_tgt_rigid = torch.zeros_like(mask_src_rigid)
                    for m in mask_tgt_rigid:
                        mask_any_tgt_rigid |= m
                    
                    mask_update_rigid = mask_src_rigid & mask_any_tgt_rigid  # [Nr]
                    idx_tgt_rigid = [torch.nonzero(m, as_tuple=False).squeeze(1) for m in mask_tgt_rigid]  # list([Nt])
                    idx_src_rigid = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)  # [Ns]
            # ===== 结束：mask 预计算 =====
            
            feat_bg, feat_rigid, rigid_visible_mask, rigid_in_crop_mask = self._build_3d_feature_volume(
                node_state_bg=node_state_bg,
                node_state_rigid=node_state_rigid,
                source_frame_idx=source_frame_idx,
                mask_src_rigid=mask_src_rigid,  # 传递 mask
                idx_src_rigid=idx_src_rigid,  # 传递索引
            )
            
            # #region agent log
            if torch.cuda.is_available():
                _debug_log(
                    "streetforward.py:train_iter",
                    "After _build_3d_feature_volume, before _predict_offsets",
                    {
                        "inner_iter": inner_iter_idx,
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                        "feat_bg_size_mb": feat_bg.numel() * 4 / 1024**2,
                        "feat_rigid_size_mb": feat_rigid.numel() * 4 / 1024**2 if feat_rigid.numel() > 0 else 0,
                    },
                    hypothesis_id="H5",
                )
            # #endregion

            feat_bg_input = feat_bg
            feat_rigid_input = feat_rigid
            feat_distant_input = None
            feat_2d_bg = None
            feat_2d_rigid = None
            feat_2d_distant = None
            if self.use_2d_features:
                feat_2d_bg, feat_2d_rigid, feat_2d_distant = self._compute_2d_features_all(
                    node_state_bg=node_state_bg,
                    node_state_rigid=node_state_rigid,
                    node_state_distant=node_state_distant,
                    source_views=batch.get("source_views", []),
                    source_images=batch.get("src_images", []),
                    source_frame_idx=source_frame_idx,
                    rigid_visible_mask=rigid_visible_mask,
                )
                if feat_2d_bg is not None and feat_bg.shape[0] == feat_2d_bg.shape[0]:
                    vis_bg = torch.ones(feat_bg.shape[0], device=self.device)
                    feat_bg_input = self._fuse_features(feat_bg, feat_2d_bg, vis_bg)
                if (
                    node_state_rigid is not None
                    and feat_rigid.shape[0] > 0
                    and feat_2d_rigid is not None
                    and feat_2d_rigid.shape[0] == feat_rigid.shape[0]
                ):
                    vis_rigid = rigid_visible_mask.float() if rigid_visible_mask is not None else torch.ones(feat_rigid.shape[0], device=self.device)
                    feat_rigid_input = self._fuse_features(feat_rigid, feat_2d_rigid, vis_rigid)
                if node_state_distant is not None and feat_2d_distant is not None:
                    zeros_3d = torch.zeros(feat_2d_distant.shape[0], self.feat_3d_dim, device=self.device)
                    vis_distant = torch.ones(feat_2d_distant.shape[0], device=self.device)
                    feat_distant_input = self._fuse_features(zeros_3d, feat_2d_distant, vis_distant)

            # Store feature tensors for baseline recording (value alignment)
            self._last_feat_3d_bg = feat_bg
            self._last_feat_3d_rigid = feat_rigid
            self._last_feat_3d_distant = feat_distant_input[:, : self.feat_3d_dim].detach().clone() if feat_distant_input is not None and feat_distant_input.numel() > 0 else None
            self._last_feat_2d_bg = feat_2d_bg
            self._last_feat_2d_rigid = feat_2d_rigid
            self._last_feat_2d_distant = feat_2d_distant
            self._last_feat_bg_input = feat_bg_input
            self._last_feat_rigid_input = feat_rigid_input
            self._last_feat_distant_input = feat_distant_input

            # #region agent log
            _debug_log(
                "streetforward.py:train_iter",
                "After 2D/3D fusion before _predict_offsets",
                {
                    "use_2d_features": bool(self.use_2d_features),
                    "feat_bg_requires_grad": bool(feat_bg.requires_grad),
                    "feat_bg_input_requires_grad": bool(feat_bg_input.requires_grad),
                    "feat_rigid_requires_grad": bool(feat_rigid.requires_grad),
                    "feat_rigid_input_requires_grad": bool(feat_rigid_input.requires_grad),
                    "feat_bg_input_shape": list(feat_bg_input.shape),
                    "feat_rigid_input_shape": list(feat_rigid_input.shape),
                    "feat_2d_bg_present": feat_2d_bg is not None,
                    "feat_2d_rigid_present": feat_2d_rigid is not None,
                },
                hypothesis_id="H4",
            )
            # #endregion
            
            offsets_bg = self._predict_offsets(feat_bg_input)
            offsets_rigid_world = None
            if node_state_rigid is not None and feat_rigid_input.shape[0] > 0:
                offsets_rigid_world = self._predict_offsets(feat_rigid_input)
                # 使用 mask_update_rigid 进行 gate（方案 1：最小改动）
                if mask_update_rigid is not None:
                    gate = mask_update_rigid.to(offsets_rigid_world["offset_pos"].dtype).unsqueeze(-1).detach()  # [Nr,1]
                    offsets_rigid_world["offset_pos"] = offsets_rigid_world["offset_pos"] * gate
                    offsets_rigid_world["offset_scales"] = offsets_rigid_world["offset_scales"] * gate
                    offsets_rigid_world["offset_quat"] = offsets_rigid_world["offset_quat"] * gate  # [Nr,4]
                    offsets_rigid_world["offset_opacity"] = offsets_rigid_world["offset_opacity"] * gate
                    offsets_rigid_world["offset_sh"] = offsets_rigid_world["offset_sh"] * gate  # [Nr, C]
                else:
                    # 回退到旧的逻辑（兼容性）
                    offsets_rigid_world = self._mask_rigid_offsets(offsets_rigid_world, rigid_visible_mask)
            offsets_distant = None
            if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
                offsets_distant = self._predict_offsets(feat_distant_input)
            
            # #region agent log
            if torch.cuda.is_available():
                _debug_log(
                    "streetforward.py:train_iter",
                    "After _predict_offsets, before _render_params_from_offsets",
                    {
                        "inner_iter": inner_iter_idx,
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    },
                    hypothesis_id="H5",
                )
            # #endregion

            # Store offsets for gradient checking (avoid accessing non-leaf tensor grads)
            self._last_offsets_bg = offsets_bg
            self._last_offsets_rigid = offsets_rigid_world
            self._last_offsets_distant = offsets_distant

            # ===== Sanity checks（方案 1：最小改动） =====
            if mask_update_rigid is not None and offsets_rigid_world is not None:
                # A) 没有监督的 rigid 点，offset 应该被 gate 成 0
                offset_pos_gated = offsets_rigid_world["offset_pos"][~mask_update_rigid]
                if offset_pos_gated.numel() > 0:
                    max_gated = offset_pos_gated.abs().max().item()
                    if max_gated > 1e-6:
                        import warnings
                        warnings.warn(
                            f"[Sanity Check A] Offsets for points without supervision should be gated to 0, "
                            f"but max abs value is {max_gated:.2e}. This may indicate a bug."
                        )
            
            # ===== 结束：Sanity checks =====
            
            render_params_bg = self._render_params_from_offsets(node_state_bg, offsets_bg)
            
            # #region agent log
            _debug_log(
                "streetforward.py:train_iter",
                "After _render_params_from_offsets for bg",
                {
                    "inner_iter": inner_iter_idx,
                    "offsets_bg_offset_pos_requires_grad": bool(offsets_bg["offset_pos"].requires_grad),
                    "offsets_bg_offset_pos_grad_fn": str(offsets_bg["offset_pos"].grad_fn)[:60] if offsets_bg["offset_pos"].grad_fn else "None",
                    "render_params_bg_means_r_requires_grad": bool(render_params_bg["means_r"].requires_grad),
                    "render_params_bg_means_r_grad_fn": str(render_params_bg["means_r"].grad_fn)[:60] if render_params_bg["means_r"].grad_fn else "None",
                    "render_params_bg_scales_r_requires_grad": bool(render_params_bg["scales_r"].requires_grad),
                    "render_params_bg_colors_r_requires_grad": bool(render_params_bg["colors_r"].requires_grad),
                },
                hypothesis_id="H6",
            )
            # #endregion
            render_params_rigid = None
            if node_state_rigid is not None and offsets_rigid_world is not None:
                # 将世界坐标的偏移量变换到局部坐标
                offsets_rigid_local = self._transform_offsets_world_to_local(
                    node_state_rigid, offsets_rigid_world, source_frame_idx
                )
                render_params_rigid = self._render_params_from_offsets(node_state_rigid, offsets_rigid_local)
            render_params_distant = None
            if offsets_distant is not None and node_state_distant is not None:
                render_params_distant = self._render_params_from_offsets(node_state_distant, offsets_distant)

            proxies_bg = self._create_proxy_params(render_params_bg)
            proxies_rigid = self._create_proxy_params(render_params_rigid) if render_params_rigid is not None else None
            proxies_distant = self._create_proxy_params(render_params_distant) if render_params_distant is not None else None

            # #region agent log
            _debug_log(
                "streetforward.py:train_iter",
                "Before target views loop",
                {
                    "num_targets": len(targets),
                    "has_rigid": proxies_rigid is not None,
                    "inner_iter": inner_iter_idx,
                },
                hypothesis_id="H1",
            )
            # #endregion

            for view_idx, target in enumerate(targets):
                view = target["view"]
                gt_img = target["gt_image"]
                target_frame_idx = int(target.get("frame_idx", source_frame_idx))
                height, width = gt_img.shape[0], gt_img.shape[1]
                num_sh = _num_sh_bases(self.sh_degree)
                resolved_frame_idx = None
                if node_state_rigid is not None:
                    node_state_rigid.cur_frame = target_frame_idx
                    resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, target_frame_idx)
                if proxies_rigid is not None and node_state_rigid is not None:
                    # ===== 方案 1：使用 idx_tgt_rigid[k] 子集索引 =====
                    if view_idx < len(idx_tgt_rigid) and len(idx_tgt_rigid[view_idx]) > 0:
                        idx = idx_tgt_rigid[view_idx]  # [Nt]
                        
                        # 1) 把 rigid proxies（local）先取子集再 transform 到 target world
                        rigid_means_local_subset = proxies_rigid["means_p"][idx]  # [Nt, 3]
                        rigid_quats_local_subset = proxies_rigid["quats_p"][idx]  # [Nt, 4]
                        rigid_scales_subset = proxies_rigid["scales_p"][idx]  # [Nt, 3]
                        rigid_opacity_subset = proxies_rigid["opacities_p"][idx]  # [Nt]
                        rigid_sh_subset = proxies_rigid["colors_p"][idx]  # [Nt, num_sh, 3]
                        
                        # 2) Transform 到 target 帧的世界坐标
                        # 传入 idx 作为 point_indices，因为 rigid_means_local_subset 是通过 idx 索引得到的子集
                        means_rigid_world = self._transform_rigid_to_world(node_state_rigid, rigid_means_local_subset, point_indices=idx)
                        quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, rigid_quats_local_subset, point_indices=idx)
                        opacities_rigid = rigid_opacity_subset
                    else:
                        # 如果没有可见点，创建空 tensor
                        means_rigid_world = torch.empty(0, 3, device=self.device)
                        quats_rigid_world = torch.empty(0, 4, device=self.device)
                        opacities_rigid = None
                        rigid_scales_subset = torch.empty(0, 3, device=self.device)
                        rigid_sh_subset = torch.empty(0, num_sh, 3, device=self.device)
                    # ===== 结束：子集索引 =====
                    
                    # #region agent log
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After rigid transform for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "means_rigid_world_requires_grad": means_rigid_world.requires_grad,
                            "means_rigid_world_grad_fn": str(means_rigid_world.grad_fn),
                            "quats_rigid_world_requires_grad": quats_rigid_world.requires_grad,
                            "num_visible_rigid": len(idx_tgt_rigid[view_idx]) if view_idx < len(idx_tgt_rigid) else 0,
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                else:
                    means_rigid_world = torch.empty(0, 3, device=self.device)
                    quats_rigid_world = torch.empty(0, 4, device=self.device)
                    opacities_rigid = None
                    rigid_scales_subset = torch.empty(0, 3, device=self.device)
                    rigid_sh_subset = torch.empty(0, num_sh, 3, device=self.device)
                
                # 合并参数（使用子集）
                if proxies_rigid is not None and node_state_rigid is not None:
                    if len(means_rigid_world) > 0:
                        # 有可见的 rigid 点，使用子集合并
                        merged_means = torch.cat([
                            proxies_bg["means_p"], 
                            means_rigid_world, 
                            proxies_distant["means_p"] if proxies_distant is not None and proxies_distant["means_p"].numel() > 0 else torch.empty(0, 3, device=self.device)
                        ], dim=0)
                        merged_quats = torch.cat([
                            proxies_bg["quats_p"], 
                            quats_rigid_world, 
                            proxies_distant["quats_p"] if proxies_distant is not None and proxies_distant["quats_p"].numel() > 0 else torch.empty(0, 4, device=self.device)
                        ], dim=0)
                        merged_scales = torch.cat([
                            proxies_bg["scales_p"], 
                            rigid_scales_subset, 
                            proxies_distant["scales_p"] if proxies_distant is not None and proxies_distant["scales_p"].numel() > 0 else torch.empty(0, 3, device=self.device)
                        ], dim=0)
                        merged_opacities = torch.cat([
                            proxies_bg["opacities_p"], 
                            opacities_rigid, 
                            proxies_distant["opacities_p"] if proxies_distant is not None and proxies_distant["opacities_p"].numel() > 0 else torch.empty(0, device=self.device)
                        ], dim=0)
                        merged_colors = torch.cat([
                            proxies_bg["colors_p"], 
                            rigid_sh_subset, 
                            proxies_distant["colors_p"] if proxies_distant is not None and proxies_distant["colors_p"].numel() > 0 else torch.empty(0, num_sh, 3, device=self.device)
                        ], dim=0)
                    else:
                        # 没有可见的 rigid 点，使用 _merge_all_params（传入 None）
                        merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = self._merge_all_params(
                            proxies_bg=proxies_bg,
                            proxies_rigid=None,
                            proxies_distant=proxies_distant,
                            means_rigid_world=means_rigid_world,
                            quats_rigid_world=quats_rigid_world,
                            opacities_rigid=opacities_rigid,
                        )
                else:
                    # 没有 rigid proxies，使用 _merge_all_params
                    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = self._merge_all_params(
                        proxies_bg=proxies_bg,
                        proxies_rigid=proxies_rigid,
                        proxies_distant=proxies_distant,
                        means_rigid_world=means_rigid_world,
                        quats_rigid_world=quats_rigid_world,
                        opacities_rigid=opacities_rigid,
                    )
                merged_params = {
                    "means_p": merged_means,
                    "scales_p": merged_scales,
                    "quats_p": merged_quats,
                    "opacities_p": merged_opacities,
                    "colors_p": merged_colors,
                }
                # #region agent log
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before _render_single_view for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "merged_means_size_mb": merged_means.numel() * 4 / 1024**2,
                            "merged_colors_size_mb": merged_colors.numel() * 4 / 1024**2,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
                
                rgb, acc = self._render_single_view(merged_params, view, height, width)
                
                # #region agent log
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After _render_single_view for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "rgb_size_mb": rgb.numel() * 4 / 1024**2,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
                
                loss = self.compute_loss(rgb, gt_img) / view_count
                total_loss_val += float(loss.detach())  # Accumulate scalar to avoid keeping graph
                
                # #region agent log
                # Check proxy gradients before backward (only log every 3 views to reduce overhead)
                if view_idx % 3 == 0 or view_idx == len(targets) - 1:
                    bg_grad_norms_before = {}
                    rigid_grad_norms_before = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        if proxies_bg[key].grad is not None:
                            bg_grad_norms_before[key] = float(proxies_bg[key].grad.norm().item())
                        else:
                            bg_grad_norms_before[key] = 0.0
                        if proxies_rigid is not None and proxies_rigid[key].grad is not None:
                            rigid_grad_norms_before[key] = float(proxies_rigid[key].grad.norm().item())
                        elif proxies_rigid is not None:
                            rigid_grad_norms_before[key] = 0.0
                    
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before backward for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "target_frame_idx": target_frame_idx,
                            "loss_value": float(loss.item()),
                            "bg_grad_norms": bg_grad_norms_before,
                            "rigid_grad_norms": rigid_grad_norms_before if proxies_rigid else None,
                        },
                        hypothesis_id="H1",
                    )
                # #endregion
                
                loss.backward()
                
                # ===== Sanity checks（方案 1：最小改动） =====
                if mask_any_tgt_rigid is not None and proxies_rigid is not None:
                    # B) 没参与任何 target 渲染的 rigid 点，proxy.grad 应该接近 0
                    if proxies_rigid["means_p"].grad is not None:
                        grad_means_not_rendered = proxies_rigid["means_p"].grad[~mask_any_tgt_rigid]
                        if grad_means_not_rendered.numel() > 0:
                            max_grad_not_rendered = grad_means_not_rendered.abs().max().item()
                            if max_grad_not_rendered > 1e-6:
                                import warnings
                                warnings.warn(
                                    f"[Sanity Check B] Gradients for points not rendered in any target should be 0, "
                                    f"but max abs value is {max_grad_not_rendered:.2e}. This may indicate a bug."
                                )
                
                # C) 每个 target 的 idx 数量合理（别全空/全满）
                if view_idx < len(idx_tgt_rigid):
                    num_visible = len(idx_tgt_rigid[view_idx])
                    num_total = node_state_rigid.means.shape[0] if node_state_rigid is not None else 0
                    if num_total > 0 and num_visible == 0:
                        import warnings
                        warnings.warn(
                            f"[Sanity Check C] Target {view_idx} has no visible rigid points "
                            f"({num_visible}/{num_total}). This may indicate a visibility issue."
                        )
                # ===== 结束：Sanity checks =====
                
                # #region agent log
                # Check proxy gradients after backward (only log every 3 views to reduce overhead)
                if view_idx % 3 == 0 or view_idx == len(targets) - 1:
                    bg_grad_norms_after = {}
                    rigid_grad_norms_after = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        if proxies_bg[key].grad is not None:
                            bg_grad_norms_after[key] = float(proxies_bg[key].grad.norm().item())
                        else:
                            bg_grad_norms_after[key] = 0.0
                        if proxies_rigid is not None and proxies_rigid[key].grad is not None:
                            rigid_grad_norms_after[key] = float(proxies_rigid[key].grad.norm().item())
                        elif proxies_rigid is not None:
                            rigid_grad_norms_after[key] = 0.0
                    
                    # Check if gradients accumulated
                    bg_grad_accumulated = {}
                    rigid_grad_accumulated = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        bg_grad_accumulated[key] = bg_grad_norms_after[key] > bg_grad_norms_before.get(key, 0.0) if view_idx > 0 else True
                        if proxies_rigid is not None:
                            rigid_grad_accumulated[key] = rigid_grad_norms_after[key] > rigid_grad_norms_before.get(key, 0.0) if view_idx > 0 else True
                    
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After backward for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "bg_grad_norms": bg_grad_norms_after,
                            "rigid_grad_norms": rigid_grad_norms_after if proxies_rigid else None,
                            "bg_grad_accumulated": bg_grad_accumulated,
                            "rigid_grad_accumulated": rigid_grad_accumulated if proxies_rigid else None,
                        },
                        hypothesis_id="H1",
                    )
                
                # Check GPU memory (only log every 5 views to reduce overhead)
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"GPU memory after view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H4",
                    )
                # #endregion
                
                # Only store images if explicitly requested (to save GPU memory)
                if self.log_images:
                    outputs.append({
                        "rgb": rgb.detach().cpu(),
                        "acc": acc.detach().cpu(),
                        "loss": loss.detach().item(),
                    })
                else:
                    outputs.append({"loss": loss.detach().item()})

            grad_report: Dict[str, float] = {}
            grad_warned = getattr(self, "_proxy_grad_warned", set())

            def _grad_or_zero(proxy_tensor: torch.Tensor, name: str) -> torch.Tensor:
                grad = proxy_tensor.grad
                if grad is None:
                    if name not in grad_warned:
                        logger.warning(f"Proxy gradient for {name} is None; using zeros for backward.")
                        grad_warned.add(name)
                    grad_report[name] = 0.0
                    return torch.zeros_like(proxy_tensor)
                grad_report[name] = float(grad.norm().detach())
                return grad

            render_tensors = [
                render_params_bg["means_r"],
                render_params_bg["scales_r"],
                render_params_bg["quats_r"],
                render_params_bg["opacities_r"],
                render_params_bg["colors_r"],
            ]
            grad_tensors = [
                _grad_or_zero(proxies_bg["means_p"], "bg.means"),
                _grad_or_zero(proxies_bg["scales_p"], "bg.scales"),
                _grad_or_zero(proxies_bg["quats_p"], "bg.quats"),
                _grad_or_zero(proxies_bg["opacities_p"], "bg.opacities"),
                _grad_or_zero(proxies_bg["colors_p"], "bg.colors"),
            ]

            if render_params_rigid is not None and proxies_rigid is not None:
                render_tensors += [
                    render_params_rigid["means_r"],
                    render_params_rigid["scales_r"],
                    render_params_rigid["quats_r"],
                    render_params_rigid["opacities_r"],
                    render_params_rigid["colors_r"],
                ]
                grad_tensors += [
                    _grad_or_zero(proxies_rigid["means_p"], "rigid.means"),
                    _grad_or_zero(proxies_rigid["scales_p"], "rigid.scales"),
                    _grad_or_zero(proxies_rigid["quats_p"], "rigid.quats"),
                    _grad_or_zero(proxies_rigid["opacities_p"], "rigid.opacities"),
                    _grad_or_zero(proxies_rigid["colors_p"], "rigid.colors"),
                ]
            if render_params_distant is not None and proxies_distant is not None:
                render_tensors += [
                    render_params_distant["means_r"],
                    render_params_distant["scales_r"],
                    render_params_distant["quats_r"],
                    render_params_distant["opacities_r"],
                    render_params_distant["colors_r"],
                ]
                grad_tensors += [
                    _grad_or_zero(proxies_distant["means_p"], "distant.means"),
                    _grad_or_zero(proxies_distant["scales_p"], "distant.scales"),
                    _grad_or_zero(proxies_distant["quats_p"], "distant.quats"),
                    _grad_or_zero(proxies_distant["opacities_p"], "distant.opacities"),
                    _grad_or_zero(proxies_distant["colors_p"], "distant.colors"),
                ]

            self._proxy_grad_warned = grad_warned
            self._last_proxy_grad_norms = grad_report
            
            # #region agent log
            # Check render params gradients before autograd.backward (only for leaf tensors)
            render_params_grad_before = {}
            for key in ["means_r", "scales_r", "quats_r", "opacities_r", "colors_r"]:
                # Only check grad for leaf tensors to avoid warnings
                if render_params_bg[key].is_leaf and render_params_bg[key].grad is not None:
                    render_params_grad_before[f"bg.{key}"] = float(render_params_bg[key].grad.norm().item())
                else:
                    render_params_grad_before[f"bg.{key}"] = 0.0
                if render_params_rigid is not None:
                    if render_params_rigid[key].is_leaf and render_params_rigid[key].grad is not None:
                        render_params_grad_before[f"rigid.{key}"] = float(render_params_rigid[key].grad.norm().item())
                    else:
                        render_params_grad_before[f"rigid.{key}"] = 0.0
                if render_params_distant is not None:
                    if render_params_distant[key].is_leaf and render_params_distant[key].grad is not None:
                        render_params_grad_before[f"distant.{key}"] = float(render_params_distant[key].grad.norm().item())
                    else:
                        render_params_grad_before[f"distant.{key}"] = 0.0
                else:
                    render_params_grad_before[f"distant.{key}"] = 0.0
            
            # Check render_tensors requires_grad status
            render_tensors_requires_grad = {}
            render_tensors_grad_fn = {}
            for i, (name, t) in enumerate(zip(
                ["bg.means_r", "bg.scales_r", "bg.quats_r", "bg.opacities_r", "bg.colors_r"],
                render_tensors[:5]
            )):
                render_tensors_requires_grad[name] = bool(t.requires_grad)
                render_tensors_grad_fn[name] = str(t.grad_fn)[:50] if t.grad_fn else "None"
            
            _debug_log(
                "streetforward.py:train_iter",
                "Before autograd.backward",
                {
                    "inner_iter": inner_iter_idx,
                    "proxy_grad_norms": grad_report,
                    "render_params_grad_before": render_params_grad_before,
                    "render_tensors_requires_grad": render_tensors_requires_grad,
                    "render_tensors_grad_fn": render_tensors_grad_fn,
                    "grad_tensors_norms": {
                        "bg.means": float(grad_tensors[0].norm().item()) if grad_tensors[0] is not None else 0.0,
                        "bg.scales": float(grad_tensors[1].norm().item()) if grad_tensors[1] is not None else 0.0,
                        "bg.quats": float(grad_tensors[2].norm().item()) if grad_tensors[2] is not None else 0.0,
                        "bg.opacities": float(grad_tensors[3].norm().item()) if grad_tensors[3] is not None else 0.0,
                        "bg.colors": float(grad_tensors[4].norm().item()) if grad_tensors[4] is not None else 0.0,
                    },
                },
                hypothesis_id="H3",
            )
            # #endregion
            
            torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)
            
            # #region agent log
            # Check render params gradients after autograd.backward (only for leaf tensors)
            # Note: render_params are non-leaf tensors, so we check the underlying computation graph
            # by checking if the offsets have gradients instead
            render_params_grad_after = {}
            # Check if offsets have gradients (these are the actual leaf tensors)
            offset_keys = ["offset_pos", "offset_scales", "offset_quat", "offset_opacity", "offset_sh"]
            offset_grads = {}
            if hasattr(self, '_last_offsets_bg') and self._last_offsets_bg is not None:
                for key in offset_keys:
                    if key in self._last_offsets_bg and self._last_offsets_bg[key].grad is not None:
                        offset_grads[f"bg.{key}"] = float(self._last_offsets_bg[key].grad.norm().item())
                    else:
                        offset_grads[f"bg.{key}"] = 0.0
            if hasattr(self, '_last_offsets_rigid') and self._last_offsets_rigid is not None:
                for key in offset_keys:
                    if key in self._last_offsets_rigid and self._last_offsets_rigid[key].grad is not None:
                        offset_grads[f"rigid.{key}"] = float(self._last_offsets_rigid[key].grad.norm().item())
                    else:
                        offset_grads[f"rigid.{key}"] = 0.0
            
            # Check MLP parameter gradients
            mlp_param_grads = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    mlp_param_grads[name] = float(param.grad.norm().item())
                else:
                    mlp_param_grads[name] = 0.0
            # Only log a few key MLP parameters to avoid log bloat
            key_params = ["mlp_offset_pos.0.weight", "mlp_conv.0.weight", "gaussion_decoder.0.weight"]
            key_mlp_grads = {k: mlp_param_grads.get(k, 0.0) for k in key_params}
            
            _debug_log(
                "streetforward.py:train_iter",
                "After autograd.backward",
                {
                    "inner_iter": inner_iter_idx,
                    "offset_grads": offset_grads,
                    "grad_propagated": {k: offset_grads[k] > 0 for k in offset_grads},
                    "key_mlp_param_grads": key_mlp_grads,
                    "any_mlp_grad": any(v > 0 for v in mlp_param_grads.values()),
                },
                hypothesis_id="H3",
            )
            # #endregion
            
            # #region agent log
            if apply_update:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if update_state:
                with torch.no_grad():
                    # Clamp means only when writing back to node_state (not during backprop)
                    means_clamped = torch.clamp(
                        render_params_bg["means_r"].detach(), min=self.bbx_min, max=self.bbx_max
                    )
                    node_state_bg.means.copy_(means_clamped)
                    node_state_bg.scales_log.copy_(render_params_bg["scales_log_r"].detach())
                    node_state_bg.quats.copy_(render_params_bg["quats_r"].detach())
                    node_state_bg.opacity_logit.copy_(render_params_bg["opacity_logit_r"].detach())
                    node_state_bg.sh_dc.copy_(render_params_bg["sh_dc_r"].detach())
                    node_state_bg.sh_rest.copy_(render_params_bg["sh_rest_r"].detach())
                    if node_state_rigid is not None and render_params_rigid is not None:
                        node_state_rigid.means.copy_(render_params_rigid["means_r"].detach())
                        node_state_rigid.scales_log.copy_(render_params_rigid["scales_log_r"].detach())
                        node_state_rigid.quats.copy_(render_params_rigid["quats_r"].detach())
                        node_state_rigid.opacity_logit.copy_(render_params_rigid["opacity_logit_r"].detach())
                        node_state_rigid.sh_dc.copy_(render_params_rigid["sh_dc_r"].detach())
                        node_state_rigid.sh_rest.copy_(render_params_rigid["sh_rest_r"].detach())
                    if node_state_distant is not None and render_params_distant is not None:
                        means_distant = torch.clamp(
                            render_params_distant["means_r"].detach(),
                            min=self.input_aabb_min,
                            max=self.input_aabb_max,
                        )
                        node_state_distant.means.copy_(means_distant)
                        node_state_distant.scales_log.copy_(render_params_distant["scales_log_r"].detach())
                        node_state_distant.quats.copy_(render_params_distant["quats_r"].detach())
                        node_state_distant.opacity_logit.copy_(render_params_distant["opacity_logit_r"].detach())
                        node_state_distant.sh_dc.copy_(render_params_distant["sh_dc_r"].detach())
                        node_state_distant.sh_rest.copy_(render_params_distant["sh_rest_r"].detach())

        self.node_states[key] = node_state_bg.detach_clone()
        if node_state_rigid is not None:
            self.node_states_rigid[key] = node_state_rigid.detach_clone()
        else:
            self.node_states_rigid[key] = None
        if node_state_distant is not None:
            self.node_states_distant[key] = node_state_distant.detach_clone()
        else:
            self.node_states_distant[key] = None
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
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []

        for view, gt_img in zip(test_views, test_images):
            height, width = gt_img.shape[0], gt_img.shape[1]
            rgb_pred, _ = self._render_single_view(render_params, view, height, width)
            rgb_gt = gt_img.to(self.device)

            psnr_list.append(self._compute_psnr(rgb_pred, rgb_gt))
            ssim_list.append(self._compute_ssim(rgb_pred, rgb_gt))
            lpips_list.append(self._compute_lpips(rgb_pred, rgb_gt))

        if len(psnr_list) == 0:
            return None

        metrics = {
            "psnr": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
            "lpips": float(np.mean(lpips_list)),
            "num_test_views": len(psnr_list),
        }
        return metrics

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

    def _compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        计算 PSNR（峰值信噪比）。
        
        Args:
            pred: 预测图像，形状 [H, W, 3]
            gt: 真实图像，形状 [H, W, 3]
            
        Returns:
            PSNR 值（dB），公式：-10 * log10(MSE)
            如果 MSE <= 0，返回 inf
        """
        mse = torch.mean((pred - gt) ** 2)
        mse_val = float(mse.item())
        if mse_val <= 0:
            return float("inf")
        psnr = -10 * torch.log10(torch.tensor(mse_val, device=pred.device))
        return float(psnr.item())

    def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        计算 SSIM（结构相似性指数）。
        
        Args:
            pred: 预测图像，形状 [H, W, 3]
            gt: 真实图像，形状 [H, W, 3]
            
        Returns:
            SSIM 值（范围 0-1），值越高越好
            如果 pytorch_msssim 不可用，返回 NaN
        """
        try:
            from pytorch_msssim import ssim
        except ImportError:
            if not getattr(self, "_ssim_unavailable", False):
                logger.warning("pytorch_msssim not installed; returning NaN for SSIM")
                self._ssim_unavailable = True
            return float("nan")

        pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
        return float(ssim(pred_4d, gt_4d, data_range=1.0).item())

    def _compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        计算 LPIPS（学习感知图像块相似性）。
        
        Args:
            pred: 预测图像，形状 [H, W, 3]
            gt: 真实图像，形状 [H, W, 3]
            
        Returns:
            LPIPS 值（通常范围 0-1），值越低越好
            如果 lpips 库不可用，返回 NaN
        
        使用 AlexNet 作为特征提取器。
        """
        try:
            from lpips import LPIPS
        except ImportError:
            if not getattr(self, "_lpips_unavailable", False):
                logger.warning("lpips not installed; returning NaN for LPIPS")
                self._lpips_unavailable = True
            return float("nan")

        if not hasattr(self, "_lpips_model") or self._lpips_model is None:
            self._lpips_model = LPIPS(net="alex").to(self.device)
        pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
        return float(self._lpips_model(pred_4d, gt_4d).item())

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
