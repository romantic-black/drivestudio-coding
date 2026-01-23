"""
StreetForward 训练器：实现基于代理参数的多视角梯度累积的前馈式 3D Gaussian Splatting 训练。

本实现参考 docs/FeedForward_3DGS_Design.md 和 docs/trainers/StreetForward_Flow.md。

核心设计：
- 将 node_state 作为分离的缓冲区（detached buffers）维护
- 使用 MLP 头预测参数偏移量（offsets）
- 通过代理参数进行渲染，实现多视角梯度累积
- 每个迭代只进行一次反向传播
- 支持静态背景和动态物体的联合训练

当外部依赖（gsplat, nerfstudio）不可用时，提供回退实现以便单元测试可以验证梯度流。
"""

from __future__ import annotations

import logging
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
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
try:
    from sklearn.neighbors import NearestNeighbors
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

logger = logging.getLogger(__name__)

# Debug logging configuration
_DEBUG_LOG_PATH = "/root/drivestudio-coding/.cursor/debug.log"

def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None, run_id: str = "initial"):
    """
    以 NDJSON 格式写入调试日志条目。
    
    Args:
        location: 日志位置标识（通常是函数名）
        message: 日志消息
        data: 日志数据字典
        hypothesis_id: 假设ID（可选）
        run_id: 运行ID（默认为 "initial"）
    """
    try:
        entry = {
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "sessionId": "debug-session",
            "runId": run_id,
        }
        if hypothesis_id:
            entry["hypothesisId"] = hypothesis_id
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write debug log: {e}")


def _num_sh_bases(degree: int) -> int:
    """
    计算球谐函数（Spherical Harmonics）基函数的数量。
    
    Args:
        degree: SH 度数
        
    Returns:
        基函数数量，公式为 (degree + 1)²
    """
    return (degree + 1) ** 2


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    将 RGB 颜色转换为球谐函数的 DC（直流）分量。
    
    Args:
        rgb: RGB 颜色张量，形状 [N, 3]，值域 [0, 1]
        
    Returns:
        SH DC 分量，形状 [N, 3]
        
    公式: sh_dc = (rgb - 0.5) / c0
    其中 c0 = 0.28209479177387814 是 SH 基函数的归一化常数
    """
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """
    将球谐函数的 DC（直流）分量转换为 RGB 颜色。
    
    Args:
        sh: SH DC 分量，形状 [N, 3]
        
    Returns:
        RGB 颜色张量，形状 [N, 3]，值域 [0, 1]
        
    公式: rgb = sh * c0 + 0.5
    其中 c0 = 0.28209479177387814 是 SH 基函数的归一化常数
    """
    c0 = 0.28209479177387814
    return sh * c0 + 0.5


def _random_quat_tensor(num: int, device: torch.device) -> torch.Tensor:
    """
    生成随机单位四元数（wxyz 格式，与 gsplat 兼容）。
    
    Args:
        num: 需要生成的四元数数量
        device: 张量设备
        
    Returns:
        随机四元数张量，形状 [num, 4]，格式为 [w, x, y, z]
        
    使用均匀采样方法生成单位球面上的随机四元数。
    """
    u = torch.rand(num, device=device)
    v = torch.rand(num, device=device)
    w = torch.rand(num, device=device)
    x = torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v)
    y = torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v)
    z = torch.sqrt(u) * torch.sin(2 * torch.pi * w)
    ww = torch.sqrt(u) * torch.cos(2 * torch.pi * w)
    return torch.stack([ww, x, y, z], dim=-1)  # wxyz format


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    四元数乘法：计算 q1 * q2（用于组合旋转）。
    
    Args:
        q1: 第一个四元数，形状 [..., 4]，格式 [w, x, y, z]
        q2: 第二个四元数，形状 [..., 4]，格式 [w, x, y, z]
        
    Returns:
        四元数乘积，形状 [..., 4]，格式 [w, x, y, z]
        
    注意：四元数乘法不满足交换律，q1 * q2 ≠ q2 * q1
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    计算四元数的共轭（用于表示逆旋转）。
    
    Args:
        q: 四元数，形状 [..., 4]，格式 [w, x, y, z]
        
    Returns:
        四元数共轭，形状 [..., 4]，格式 [w, -x, -y, -z]
        
    对于单位四元数，共轭等于逆四元数。
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化四元数，确保为单位四元数。
    
    Args:
        q: 四元数，形状 [..., 4]
        eps: 防止除零的小常数
        
    Returns:
        归一化后的四元数，形状 [..., 4]
        
    单位四元数用于表示旋转，必须满足 ||q|| = 1
    """
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数（wxyz 格式）转换为旋转矩阵。
    
    Args:
        q: 四元数，形状 [..., 4]，格式 [w, x, y, z]
        
    Returns:
        旋转矩阵，形状 [..., 3, 3]
        
    使用标准的四元数到旋转矩阵的转换公式。
    """
    q = _normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _axis_angle_to_quat(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion (wxyz format).
    
    Uses branchless sinc structure to avoid discontinuities near threshold,
    providing smoother gradients.
    
    Args:
        omega: [N, 3] axis-angle vector
        eps: Small epsilon for numerical stability
        
    Returns:
        quat: [N, 4] quaternion in wxyz format
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [N, 1]
    half_theta = theta * 0.5
    
    # Use sinc structure: sin(θ/2) / (θ + eps), avoids division by zero and is continuously differentiable
    # When θ → 0, sinc(θ/2) → 1/2, so xyz = ω * (1/2) = ω/2 (correct small angle approximation)
    sinc_half = torch.sin(half_theta) / (theta + eps)  # [N, 1]
    xyz = omega * sinc_half  # [N, 3]
    w = torch.cos(half_theta)  # [N, 1]
    
    return torch.cat([w, xyz], dim=-1)  # [N, 4] wxyz format


def get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    将相机到世界的变换矩阵转换为世界到相机的视图矩阵（gsplat 使用的格式）。
    
    Args:
        camera_to_world: 相机到世界的变换矩阵，形状 [4, 4] 或 [B, 4, 4]
        
    Returns:
        世界到相机的视图矩阵，形状 [B, 4, 4]
        
    注意：gsplat 使用特殊的坐标系约定，需要对旋转矩阵的 Y 和 Z 轴取反。
    """
    if camera_to_world.dim() == 2:
        camera_to_world = camera_to_world.unsqueeze(0)  # [1,4,4]
    r = camera_to_world[:, :3, :3]
    t = camera_to_world[:, :3, 3:4]
    r = r * torch.tensor([[[1, -1, -1]]], device=r.device, dtype=r.dtype)
    r_inv = r.transpose(1, 2)
    t_inv = -torch.bmm(r_inv, t)
    viewmat = torch.zeros(r.shape[0], 4, 4, device=r.device, dtype=r.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = r_inv
    viewmat[:, :3, 3:4] = t_inv
    return viewmat


from gsplat.rendering import rasterization as _gsplat_rasterization


from models.evol_splat import (
    SparseCostRegNet as _SparseCostRegNet,
    construct_sparse_tensor as _construct_sparse_tensor,
    sparse_to_dense_volume as _sparse_to_dense_volume,
)


def _pairwise_neighbor_distances(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Compute k-NN distances efficiently using sklearn's NearestNeighbors.
    
    This function avoids the O(N²) memory overhead of computing full pairwise distances
    by using efficient k-NN algorithms. For large point clouds (e.g., 890K points),
    this reduces memory from ~3.17 TB to ~10.7 MB.
    
    Args:
        points: Tensor of shape [N, 3] containing point coordinates
        k: Number of nearest neighbors to find
        
    Returns:
        Tensor of shape [N, k] containing distances to k nearest neighbors
    """
    if not _sklearn_available:
        raise ImportError("sklearn is required for k-NN search. Please install scikit-learn.")
    
    # Convert to numpy (CPU) for sklearn
    if points.is_cuda:
        points_np = points.cpu().numpy().astype('float32')
    else:
        points_np = points.numpy().astype('float32')
    
    # Build the nearest neighbors model
    # Use k+1 neighbors because the point itself will be included as the first neighbor
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn_model.fit(points_np)
    
    # Find the k-nearest neighbors
    distances, _ = nn_model.kneighbors(points_np)
    
    # Exclude self (first neighbor is always the point itself)
    distances = distances[:, 1:]  # [N, k]
    
    # Convert back to torch tensor on original device
    result = torch.from_numpy(distances.astype('float32')).to(points.device)
    
    return result

@dataclass
class NodeStateBackground:
    """
    静态背景的节点状态，存储分离的高斯参数（世界坐标系）。
    
    所有参数都是分离的（detached），不参与梯度计算，作为稳定的参数缓冲区。
    每个 (scene_id, segment_id) 对应一个 NodeStateBackground。
    
    Attributes:
        means: Gaussian 中心位置，形状 [N_bg, 3]，世界坐标系
        scales_log: 尺度的对数，形状 [N_bg, 3]（3个轴）
        quats: 旋转四元数，形状 [N_bg, 4]，wxyz 格式
        opacity_logit: 不透明度的 logit 值，形状 [N_bg, 1]
        sh_dc: 球谐函数 DC 分量（RGB），形状 [N_bg, 3]
        sh_rest: 球谐函数高阶分量，形状 [N_bg, num_sh-1, 3]
    """
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateBackground":
        """
        创建节点状态的分离副本。
        
        Returns:
            新的 NodeStateBackground 实例，所有张量都是分离的副本
        """
        return NodeStateBackground(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


@dataclass
class NodeStateRigid:
    """
    动态物体的节点状态，存储分离的高斯参数（局部坐标系）。
    
    所有参数都是分离的（detached），不参与梯度计算，作为稳定的参数缓冲区。
    每个 (scene_id, segment_id) 对应一个 NodeStateRigid（如果存在动态物体）。
    
    Attributes:
        means: Gaussian 中心位置，形状 [N_rigid, 3]，局部坐标系
        scales_log: 尺度的对数，形状 [N_rigid, 3]（3个轴）
        quats: 旋转四元数，形状 [N_rigid, 4]，wxyz 格式，局部旋转
        opacity_logit: 不透明度的 logit 值，形状 [N_rigid, 1]
        sh_dc: 球谐函数 DC 分量（RGB），形状 [N_rigid, 3]
        sh_rest: 球谐函数高阶分量，形状 [N_rigid, num_sh-1, 3]
        point_ids: 每个点属于哪个实例，形状 [N_rigid, 1]
        instances_quats: 实例旋转，形状 [num_frames, num_instances, 4]，wxyz 格式
        instances_trans: 实例平移，形状 [num_frames, num_instances, 3]
        instances_fv: 实例可见性，形状 [num_frames, num_instances]，bool 类型
        instance_ids: 实例ID列表
        frame_ids: 帧ID列表（用于索引 instances_*）
        cur_frame: 当前帧索引（用于变换）
    """
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    point_ids: torch.Tensor
    instances_quats: torch.Tensor
    instances_trans: torch.Tensor
    instances_fv: torch.Tensor
    instance_ids: List[int]
    frame_ids: List[int]
    cur_frame: int

    def detach_clone(self) -> "NodeStateRigid":
        """
        创建节点状态的分离副本。
        
        Returns:
            新的 NodeStateRigid 实例，所有张量都是分离的副本
        """
        return NodeStateRigid(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
            point_ids=self.point_ids.detach().clone(),
            instances_quats=self.instances_quats.detach().clone(),
            instances_trans=self.instances_trans.detach().clone(),
            instances_fv=self.instances_fv.detach().clone(),
            instance_ids=list(self.instance_ids),
            frame_ids=list(self.frame_ids),
            cur_frame=int(self.cur_frame),
        )


@dataclass
class NodeStateDistant:
    """
    背景静态点的状态（crop_aabb 外、input_aabb 内）。

    与 NodeStateBackground 类似，使用世界坐标系，但不参与 3D 特征体积构建。
    """
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateDistant":
        """
        创建节点状态的分离副本。
        """
        return NodeStateDistant(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


NodeState = NodeStateBackground


class StreetForwardTrainer(nn.Module):
    """
    基于代理参数的多视角梯度累积的前馈式 3D Gaussian Splatting 训练器。
    
    核心设计理念：
    - 双 NodeState 架构：每个 (scene_id, segment_id) 维护两个 NodeState：
      * NodeStateBackground：存储静态背景的高斯参数（世界坐标系）
      * NodeStateRigid：存储动态物体的高斯参数（局部坐标系）
    - 前馈预测：通过 3D 特征体积预测偏移量（offsets），静态和动态物体共享相同的 MLP 网络
    - 代理参数渲染：使用代理参数进行渲染，实现多视角梯度累积
    - 单次反向传播：每个迭代只进行一次反向传播
    - 帧变换机制：动态物体在不同帧间通过 RigidNodes 变换，支持时间一致性
    
    训练流程：
    1. 获取或初始化双 NodeState（Background + RigidNodes）
    2. 构建 3D 特征体积（合并静态和动态点云）
    3. 预测偏移量（静态和动态共同预测）
    4. 计算渲染参数（分别应用到两个 NodeState）
    5. 创建代理参数（分别创建静态和动态代理）
    6. 遍历所有 target 帧，渲染并累积梯度
    7. 反向传播到渲染参数，然后自动传播到网络参数
    8. 更新双 NodeState
    
    详细流程请参考 docs/trainers/StreetForward_Flow.md
    """

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

    def _compute_initial_scales(self, means: torch.Tensor) -> torch.Tensor:
        """
        基于 k-NN 距离计算初始尺度（对数域）。
        
        Args:
            means: 点位置，形状 [N, 3]
            
        Returns:
            初始尺度对数，形状 [N, 3]
            
        方法：计算每个点到 k 个最近邻的平均距离，取对数作为初始尺度。
        使用 clamp 确保距离不小于 1e-3，避免对数域中的数值问题。
        """
        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        return torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

    def _init_node_state_from_arrays(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        state_cls,
    ):
        """
        从点和颜色数组初始化节点状态。
        """
        if len(points) == 0:
            raise ValueError("Empty point cloud provided for node state initialization.")

        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor

        initial_scales = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        return state_cls(
            means=means.detach().clone(),
            scales_log=initial_scales.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )

    def _init_node_from_pointcloud(
        self,
        scene_id: int,
        segment_id: int,
        pointcloud,
    ) -> NodeState:
        """
        从点云初始化静态背景的 NodeState。
        
        Args:
            scene_id: 场景ID
            segment_id: 片段ID
            pointcloud: 点云数据，可以是字典格式 {"background": [N, 6]} 或对象格式（需有 points 和 colors 属性）
            
        Returns:
            初始化的 NodeStateBackground
            
        处理流程：
        1. 提取点坐标和颜色（如果是字典格式，从 "background" 键获取）
        2. 将颜色归一化到 [0, 1] 范围（如果值域是 [0, 255]）
        3. 计算初始尺度（基于 k-NN 距离）
        4. 生成随机四元数
        5. 初始化不透明度为 logit(0.1)
        6. 将 RGB 转换为 SH DC 分量
        7. 初始化 SH rest 分量为零
        8. 所有参数初始化为分离状态
        """
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3]
            if background.shape[1] >= 6:
                colors = background[:, 3:]
                if colors.max() > 1.0 + 1e-3:
                    colors = colors / 255.0
            else:
                colors = np.zeros_like(points)
        else:
            points = np.asarray(pointcloud.points)  # type: ignore[attr-defined]
            colors = np.asarray(pointcloud.colors)  # type: ignore[attr-defined]
            if colors.max() > 1.0 + 1e-3:
                colors = colors / 255.0

        if len(points) == 0:
            raise ValueError(f"Empty point cloud for scene {scene_id}, segment {segment_id}")

        node_state = self._init_node_state_from_arrays(points, colors, NodeStateBackground)
        self.node_states[(scene_id, segment_id)] = node_state
        return node_state

    def _init_rigid_node_state_from_pcd(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        point_ids: torch.Tensor,
        dynamic_info: Dict,
        frame_ids: List[int],
        instance_id_map: Dict[int, int],
        instance_ids: List[int],
    ) -> NodeStateRigid:
        """
        从点云初始化动态物体的 NodeStateRigid。
        
        Args:
            points: 点坐标数组，形状 [N_rigid, 3]，局部坐标系
            colors: 颜色数组，形状 [N_rigid, 3]
            point_ids: 每个点属于哪个实例，形状 [N_rigid, 1]
            dynamic_info: 动态物体信息字典，包含各帧的实例位姿
            frame_ids: 帧ID列表
            instance_id_map: 实例ID到索引的映射
            instance_ids: 实例ID列表
            
        Returns:
            初始化的 NodeStateRigid
            
        处理流程：
        1. 将点坐标和颜色转换为张量，归一化颜色到 [0, 1]
        2. 计算初始尺度（基于 k-NN 距离）
        3. 生成随机四元数（局部旋转）
        4. 初始化不透明度为 logit(0.1)
        5. 将 RGB 转换为 SH DC 分量
        6. 初始化 SH rest 分量为零
        7. 从 dynamic_info 初始化 instances_quats、instances_trans 和 instances_fv
        8. 所有参数初始化为分离状态
        """
        means = torch.tensor(points, dtype=torch.float32, device=self.device)
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor
        scales_log = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        num_frames = len(frame_ids)
        num_instances = len(instance_id_map)
        instances_quats = torch.zeros(num_frames, num_instances, 4, device=self.device)
        instances_trans = torch.zeros(num_frames, num_instances, 3, device=self.device)
        instances_fv = torch.zeros(num_frames, num_instances, dtype=torch.bool, device=self.device)
        instances_quats[..., 0] = 1.0

        frame_id_map = {fid: idx for idx, fid in enumerate(frame_ids)}
        for frame_id, frame_info in dynamic_info.items():
            frame_idx = int(frame_id)
            if frame_idx not in frame_id_map:
                continue
            frame_slot = frame_id_map[frame_idx]
            instances = frame_info.get("instances", {})
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        continue
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=self.device)
                    trans = torch.tensor(instance_pose["trans"], device=self.device)
                    instances_quats[frame_slot, ins_slot] = quat
                    instances_trans[frame_slot, ins_slot] = trans
                    instances_fv[frame_slot, ins_slot] = True

        return NodeStateRigid(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
            point_ids=point_ids.detach().clone(),
            instances_quats=instances_quats.detach().clone(),
            instances_trans=instances_trans.detach().clone(),
            instances_fv=instances_fv.detach().clone(),
            instance_ids=list(instance_ids),
            frame_ids=list(frame_ids),
            cur_frame=0,
        )

    def _get_or_init_node_states(
        self, batch: Dict
    ) -> Tuple[Tuple[int, int], NodeState, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        """
        获取或初始化双 NodeState（Background + RigidNodes）。
        
        Args:
            batch: 批次数据字典，需包含：
                - "scene_id": 场景ID
                - "segment_id": 片段ID
                - "pointcloud": 点云数据（字典格式包含 "background" 和可选的 "dynamic"）
                - "dynamic_info": 动态物体信息（可选）
                
        Returns:
            (key, node_state_bg, node_state_rigid, node_state_distant) 元组：
                - key: (scene_id, segment_id) 元组
                - node_state_bg: NodeStateBackground（静态背景）
                - node_state_rigid: NodeStateRigid 或 None（动态物体，如果存在）
                - node_state_distant: NodeStateDistant 或 None（背景远景，如果启用）
                
        处理流程：
        1. 如果 NodeState 已存在，直接返回（支持动态扩展帧信息）
        2. 如果 NodeState 不存在（新段开始），清空所有缓存以释放显存，然后从点云初始化
        3. 如果点云包含动态物体，会同时初始化 NodeStateRigid
        
        注意：当遇到新的 (scene_id, segment_id) 时，会自动清空之前的 node_states 缓存，
        只保留当前段的状态，以节省显存。这对于顺序训练多个段的场景特别有用。
        """
        scene_id = batch["scene_id"]
        if isinstance(scene_id, torch.Tensor):
            scene_id = int(scene_id.item())
        segment_id = batch["segment_id"]
        if isinstance(segment_id, torch.Tensor):
            segment_id = int(segment_id.item())
        key = (scene_id, segment_id)
        if key in self.node_states:
            node_state_rigid = self.node_states_rigid.get(key)
            node_state_distant = self.node_states_distant.get(key)
            dynamic_info = batch.get("dynamic_info")
            if node_state_rigid is not None and dynamic_info:
                node_state_rigid = self._extend_rigid_frames(node_state_rigid, dynamic_info)
                self.node_states_rigid[key] = node_state_rigid
            return key, self.node_states[key], node_state_rigid, node_state_distant
        
        # 如果 key 不存在，说明已经开始下一个段的训练，清空之前的缓存以释放显存
        if len(self.node_states) > 0:
            logger.debug(f"Clearing node_states cache before initializing new segment {key}. Previous cache had {len(self.node_states)} entries.")
            self.node_states.clear()
            self.node_states_rigid.clear()
            self.node_states_distant.clear()
            # 强制垃圾回收以释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pointcloud = batch["pointcloud"]
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            if background.shape[1] >= 6:
                colors = background[:, 3:6].astype(np.float32)
            else:
                colors = np.zeros_like(points, dtype=np.float32)
        else:
            points = np.asarray(getattr(pointcloud, "points", np.zeros((0, 3))), dtype=np.float32)
            raw_colors = getattr(pointcloud, "colors", None)
            if raw_colors is not None:
                colors = np.asarray(raw_colors, dtype=np.float32)
                if colors.ndim == 1:
                    colors = np.expand_dims(colors, axis=0)
                if colors.shape[0] != points.shape[0]:
                    colors = np.zeros_like(points, dtype=np.float32)
            else:
                colors = np.zeros_like(points, dtype=np.float32)
        # 过滤到 input_aabb 范围内
        input_min = self.input_aabb_min.cpu().numpy()
        input_max = self.input_aabb_max.cpu().numpy()
        if points.size > 0:
            inside_mask = (
                (points >= input_min)
                & (points <= input_max)
            ).all(axis=1)
            points = points[inside_mask]
            colors = colors[inside_mask]

        crop_min = self.bbx_min.cpu().numpy()
        crop_max = self.bbx_max.cpu().numpy()
        in_crop_mask = (
            (points >= crop_min)
            & (points <= crop_max)
        ).all(axis=1)
        fg_points = points[in_crop_mask]
        fg_colors = colors[in_crop_mask]
        distant_points = points[~in_crop_mask]
        distant_colors = colors[~in_crop_mask]

        node_state_bg = self._init_node_state_from_arrays(fg_points, fg_colors, NodeStateBackground)
        node_state_distant: Optional[NodeStateDistant] = None
        if len(distant_points) > 0:
            node_state_distant = self._init_node_state_from_arrays(
                distant_points.astype(np.float32),
                distant_colors.astype(np.float32),
                NodeStateDistant,
            )
        self.node_states[(scene_id, segment_id)] = node_state_bg

        node_state_rigid: Optional[NodeStateRigid] = None
        if isinstance(pointcloud, dict) and pointcloud.get("dynamic"):
            dynamic_points = []
            dynamic_colors = []
            point_ids = []
            instance_ids = sorted(int(ins_id) for ins_id in pointcloud["dynamic"].keys())
            instance_id_map = {ins_id: idx for idx, ins_id in enumerate(instance_ids)}
            for ins_id in instance_ids:
                instance_pcd = pointcloud["dynamic"][ins_id]
                if instance_pcd is None or len(instance_pcd) == 0:
                    continue
                n_points = instance_pcd.shape[0]
                dynamic_points.append(instance_pcd[:, :3])
                dynamic_colors.append(instance_pcd[:, 3:6])
                point_ids.extend([instance_id_map[ins_id]] * n_points)

            if dynamic_points:
                dynamic_points = np.concatenate(dynamic_points, axis=0)
                dynamic_colors = np.concatenate(dynamic_colors, axis=0)
                point_ids_tensor = torch.tensor(point_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
                dynamic_info = batch.get("dynamic_info")
                if not dynamic_info:
                    raise ValueError("dynamic_info is required when dynamic pointclouds are provided.")
                frame_ids = sorted(int(fid) for fid in dynamic_info.keys())
                node_state_rigid = self._init_rigid_node_state_from_pcd(
                    points=dynamic_points,
                    colors=dynamic_colors,
                    point_ids=point_ids_tensor,
                    dynamic_info=dynamic_info,
                    frame_ids=frame_ids,
                    instance_id_map=instance_id_map,
                    instance_ids=instance_ids,
                )

        self.node_states_rigid[key] = node_state_rigid
        self.node_states_distant[key] = node_state_distant
        return key, node_state_bg, node_state_rigid, node_state_distant


    def _node_state_to_dict(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        """
        将 NodeState 转换为字典（用于保存检查点）。
        
        Args:
            node_state: NodeState 对象
            
        Returns:
            状态字典，所有张量都已分离并移到 CPU
        """
        return {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
        }

    def _node_state_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> NodeState:
        """
        从字典恢复 NodeState（用于加载检查点）。
        
        Args:
            state_dict: 状态字典
            
        Returns:
            恢复的 NodeState，所有张量都已移到设备并分离
        """
        return NodeState(
            means=state_dict["means"].to(self.device),
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
        ).detach_clone()

    def _node_state_distant_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> NodeStateDistant:
        """
        从字典恢复 NodeStateDistant（用于加载检查点）。
        """
        return NodeStateDistant(
            means=state_dict["means"].to(self.device),
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
        ).detach_clone()

    def _node_state_rigid_to_dict(self, node_state: NodeStateRigid) -> Dict:
        """
        将 NodeStateRigid 转换为字典（用于保存检查点）。
        
        Args:
            node_state: NodeStateRigid 对象
            
        Returns:
            状态字典，所有张量都已分离并移到 CPU
        """
        return {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
            "point_ids": node_state.point_ids.detach().cpu(),
            "instances_quats": node_state.instances_quats.detach().cpu(),
            "instances_trans": node_state.instances_trans.detach().cpu(),
            "instances_fv": node_state.instances_fv.detach().cpu(),
            "instance_ids": list(node_state.instance_ids),
            "frame_ids": list(node_state.frame_ids),
            "cur_frame": int(node_state.cur_frame),
        }

    def _node_state_rigid_from_dict(self, state_dict: Dict) -> NodeStateRigid:
        """
        从字典恢复 NodeStateRigid（用于加载检查点）。
        
        Args:
            state_dict: 状态字典
            
        Returns:
            恢复的 NodeStateRigid，所有张量都已移到设备并分离
        """
        instance_ids = state_dict.get("instance_ids")
        if instance_ids is None:
            num_instances = state_dict["instances_quats"].shape[1]
            instance_ids = list(range(num_instances))
        elif isinstance(instance_ids, torch.Tensor):
            instance_ids = instance_ids.tolist()
        return NodeStateRigid(
            means=state_dict["means"].to(self.device),
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
            point_ids=state_dict["point_ids"].to(self.device),
            instances_quats=state_dict["instances_quats"].to(self.device),
            instances_trans=state_dict["instances_trans"].to(self.device),
            instances_fv=state_dict["instances_fv"].to(self.device),
            instance_ids=list(instance_ids),
            frame_ids=list(state_dict.get("frame_ids", [])),
            cur_frame=int(state_dict.get("cur_frame", 0)),
        ).detach_clone()

    def _extend_rigid_frames(self, node_state_rigid: NodeStateRigid, dynamic_info: Dict) -> NodeStateRigid:
        """
        扩展 RigidNodes 的帧信息，添加新的帧数据。
        
        Args:
            node_state_rigid: 现有的 RigidNodes
            dynamic_info: 动态物体信息字典，包含新帧的实例位姿
            
        Returns:
            扩展后的 RigidNodes
        
        如果 dynamic_info 中包含新的帧ID，会将这些帧的实例位姿添加到 instances_* 张量中。
        """
        if not dynamic_info:
            return node_state_rigid
        existing_frame_ids = set(node_state_rigid.frame_ids)
        candidate_frame_ids = [int(fid) for fid in dynamic_info.keys()]
        new_frame_ids = [fid for fid in candidate_frame_ids if fid not in existing_frame_ids]
        if not new_frame_ids:
            return node_state_rigid

        new_frame_ids = sorted(new_frame_ids)
        num_new_frames = len(new_frame_ids)
        num_instances = node_state_rigid.instances_quats.shape[1]
        device = node_state_rigid.instances_quats.device

        new_quats = torch.zeros((num_new_frames, num_instances, 4), device=device)
        new_trans = torch.zeros((num_new_frames, num_instances, 3), device=device)
        new_fv = torch.zeros((num_new_frames, num_instances), dtype=torch.bool, device=device)
        new_quats[..., 0] = 1.0

        if node_state_rigid.instance_ids:
            instance_id_map = {int(ins_id): idx for idx, ins_id in enumerate(node_state_rigid.instance_ids)}
        else:
            instance_id_map = {int(idx): idx for idx in range(num_instances)}

        for frame_slot, frame_id in enumerate(new_frame_ids):
            frame_info = dynamic_info.get(frame_id)
            if frame_info is None:
                frame_info = dynamic_info.get(str(frame_id))
            if not frame_info:
                continue
            instances = frame_info.get("instances", {})
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        continue
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=device)
                    trans = torch.tensor(instance_pose["trans"], device=device)
                    new_quats[frame_slot, ins_slot] = quat
                    new_trans[frame_slot, ins_slot] = trans
                    new_fv[frame_slot, ins_slot] = True

        node_state_rigid.instances_quats = torch.cat([node_state_rigid.instances_quats, new_quats], dim=0)
        node_state_rigid.instances_trans = torch.cat([node_state_rigid.instances_trans, new_trans], dim=0)
        node_state_rigid.instances_fv = torch.cat([node_state_rigid.instances_fv, new_fv], dim=0)
        node_state_rigid.frame_ids.extend(new_frame_ids)
        return node_state_rigid

    def _resolve_rigid_frame_idx(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> int:
        """
        将 frame_idx（frame ID）解析为 frame_ids 列表中的索引。
        
        Args:
            frame_idx: 场景全局 frame ID（不是索引）
            
        Returns:
            frame_ids 列表中的索引
            
        Raises:
            ValueError: 如果 frame_idx 不在 frame_ids 列表中
        """
        if not node_state_rigid.frame_ids:
            # 如果没有 frame_ids，假设 frame_idx 就是索引
            return int(frame_idx)
        
        # 首先检查 frame_idx 是否是 frame ID
        if frame_idx in node_state_rigid.frame_ids:
            return node_state_rigid.frame_ids.index(frame_idx)
        
        # 如果找不到，抛出错误
        raise ValueError(
            f"Frame ID {frame_idx} not found in frame_ids {node_state_rigid.frame_ids}. "
            f"Please ensure the frame_idx is a valid frame ID, not an index."
        )

    def _transform_rigid_to_world(
        self, node_state_rigid: NodeStateRigid, means_local: torch.Tensor
    ) -> torch.Tensor:
        """
        将动态物体的局部坐标位置变换到世界坐标。
        
        Args:
            node_state_rigid: Rigid node state，包含实例位姿信息
            means_local: 局部坐标的位置，形状 [N_rigid, 3]（可微）
            
        Returns:
            世界坐标的位置，形状 [N_rigid, 3]（可微）
        
        变换公式：means_world = R * means_local + t
        其中 R 和 t 从 node_state_rigid.instances_* 中获取，根据 cur_frame 和 point_ids 选择。
        
        关键点：
        - 保持梯度连接，不使用 detach，让 PyTorch 自动处理梯度反向传播
        - 使用当前帧（cur_frame）的实例位姿进行变换
        """
        # #region agent log
        _debug_log(
            "streetforward.py:_transform_rigid_to_world",
            "Transforming rigid to world",
            {
                "num_points": means_local.shape[0],
                "cur_frame": node_state_rigid.cur_frame,
                "means_local_requires_grad": means_local.requires_grad,
                "means_local_is_leaf": means_local.is_leaf,
            },
            hypothesis_id="H2",
        )
        # #endregion
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        trans_cur_frame = node_state_rigid.instances_trans[frame_idx]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)
        rot_per_pts = rot_cur_frame[node_state_rigid.point_ids[..., 0]]
        trans_per_pts = trans_cur_frame[node_state_rigid.point_ids[..., 0]]
        means_world = torch.bmm(rot_per_pts, means_local.unsqueeze(-1)).squeeze(-1) + trans_per_pts
        # #region agent log
        _debug_log(
            "streetforward.py:_transform_rigid_to_world",
            "Transformation complete",
            {
                "means_world_requires_grad": means_world.requires_grad,
                "means_world_is_leaf": means_world.is_leaf,
                "grad_fn": str(means_world.grad_fn),
            },
            hypothesis_id="H2",
        )
        # #endregion
        return means_world

    def _transform_rigid_quats_to_world(
        self, node_state_rigid: NodeStateRigid, quats_local: torch.Tensor
    ) -> torch.Tensor:
        """
        将动态物体的局部坐标旋转变换到世界坐标。
        
        Args:
            node_state_rigid: Rigid node state，包含实例旋转信息
            quats_local: 局部坐标的四元数，形状 [N_rigid, 4]（可微）
            
        Returns:
            世界坐标的四元数，形状 [N_rigid, 4]（可微）
        
        变换公式：quats_world = normalize(quats_instance * quats_local)
        使用四元数乘法组合实例旋转和局部旋转。
        """
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        quats_per_pts = quats_cur_frame[node_state_rigid.point_ids[..., 0]]
        return _normalize_quat(_quat_multiply(quats_per_pts, quats_local))

    def _transform_offsets_world_to_local(
        self, node_state_rigid: NodeStateRigid, offsets_world: Dict[str, torch.Tensor], frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        将世界坐标的偏移量变换到局部坐标系。
        
        关键：offsets 是向量，变换方式与位置不同（只需要旋转，不需要平移）。
        
        Args:
            node_state_rigid: Rigid node state
            offsets_world: 世界坐标的偏移量字典
            frame_idx: 当前帧的 frame ID
            
        Returns:
            局部坐标的偏移量字典
        """
        resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        
        # 获取当前帧的旋转矩阵
        quats_cur_frame = node_state_rigid.instances_quats[resolved_frame_idx]  # [num_instances, 4]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)  # [num_instances, 3, 3]
        rot_per_pts = rot_cur_frame[node_state_rigid.point_ids[..., 0]]  # [N_rigid, 3, 3]
        
        # 将世界坐标的位置偏移量变换到局部坐标
        # 对于向量（偏移量），只需要旋转，不需要平移
        offset_pos_world = offsets_world["offset_pos"]  # [N_rigid, 3]
        offset_pos_local = torch.bmm(
            rot_per_pts.transpose(-2, -1),  # R^T: [N_rigid, 3, 3]
            offset_pos_world.unsqueeze(-1)  # [N_rigid, 3, 1]
        ).squeeze(-1)  # [N_rigid, 3]
        
        # 将世界坐标的旋转增量转换到局部坐标：q_local = q_inst^{-1} * q_world * q_inst
        offset_quat_world = offsets_world["offset_quat"]
        quats_per_pts = _normalize_quat(node_state_rigid.instances_quats[resolved_frame_idx][node_state_rigid.point_ids[..., 0]])
        quats_inv = _quat_conjugate(quats_per_pts)
        offset_quat = _normalize_quat(_quat_multiply(_quat_multiply(quats_inv, offset_quat_world), quats_per_pts))
        
        # 其他偏移量（scales, opacity, sh）是标量或颜色，不需要坐标变换
        return {
            "offset_pos": offset_pos_local,
            "offset_scales": offsets_world["offset_scales"],  # 尺度不变
            "offset_quat": offset_quat,
            "offset_opacity": offsets_world["offset_opacity"],  # 不变
            "offset_sh": offsets_world["offset_sh"],  # 不变
        }

    def _build_3d_feature_volume(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        构建 3D 特征体积，为静态背景和动态物体提取特征。
        
        这是训练流程中的核心步骤，详细流程请参考 docs/trainers/StreetForward_Flow.md。
        
        Args:
            node_state_bg: 静态背景的 NodeState（世界坐标系）
            node_state_rigid: 动态物体的 NodeStateRigid（局部坐标系），可选
            source_frame_idx: Source 帧的 frame ID（场景全局 frame_idx）
            
        Returns:
            (feat_3d_crop_bg, feat_3d_crop_rigid, rigid_visible_mask, rigid_in_crop_mask) 元组：
                - feat_3d_crop_bg: 静态背景点的3D特征，形状 [N_bg, outdim]
                - feat_3d_crop_rigid: 动态物体点的3D特征，形状 [N_rigid, outdim]
                - rigid_visible_mask: 动态物体可见性掩码，形状 [N_rigid]，可选
                - rigid_in_crop_mask: 动态物体是否在 crop_aabb 内的掩码，形状 [N_rigid]，可选
        
        处理流程：
        1. 设置 RigidNodes.cur_frame = source_frame_idx
        2. 获取静态背景点云（世界坐标）
        3. 变换动态物体到 source 帧的世界坐标
        4. 合并静态和动态点云
        5. 构建统一的 3D 特征体积（稀疏张量 → 稀疏卷积 → 密集体积）
        6. 分别为静态和动态点插值特征
        7. 删除密集体积以释放内存
        """
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Start building 3D feature volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_bg_points": node_state_bg.means.shape[0],
                    "has_rigid": node_state_rigid is not None,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        rigid_visible_mask = None
        rigid_in_crop_mask = None
        if node_state_rigid is not None:
            node_state_rigid.cur_frame = source_frame_idx
            resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, source_frame_idx)
            visibility = node_state_rigid.instances_fv[resolved_frame_idx]
            rigid_visible_mask = visibility[node_state_rigid.point_ids[..., 0]].bool()

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)

        means_rigid_world_all = torch.empty(0, 3, device=self.device)
        anchor_rgb_rigid_all = torch.empty(0, 3, device=self.device)
        if node_state_rigid is not None:
            means_rigid_world_all = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            anchor_rgb_rigid_all = _sh_to_rgb(node_state_rigid.sh_dc)
            rigid_in_crop_mask = torch.all(
                (means_rigid_world_all >= self.bbx_min) & (means_rigid_world_all <= self.bbx_max),
                dim=-1,
            )

        effective_mask = None
        if node_state_rigid is not None and means_rigid_world_all.numel() > 0:
            effective_mask = rigid_in_crop_mask
            if effective_mask is None:
                effective_mask = torch.ones(means_rigid_world_all.shape[0], dtype=torch.bool, device=self.device)
            if rigid_visible_mask is not None:
                effective_mask = effective_mask & rigid_visible_mask

        means_list = [means_bg]
        rgb_list = [anchor_rgb_bg]
        if node_state_rigid is not None and means_rigid_world_all.numel() > 0 and effective_mask is not None and effective_mask.any():
            means_list.append(means_rigid_world_all[effective_mask])
            rgb_list.append(anchor_rgb_rigid_all[effective_mask])

        means_all = torch.cat(means_list, dim=0)
        anchor_rgb_all = torch.cat(rgb_list, dim=0)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Before construct_sparse_tensor",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_total_points": means_all.shape[0],
                    "means_all_size_mb": means_all.numel() * 4 / 1024**2,
                    "anchor_rgb_all_size_mb": anchor_rgb_all.numel() * 4 / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion

        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means_all.clone(),
            feats=anchor_rgb_all,
            Bbx_max=self.bbx_max,
            Bbx_min=self.bbx_min,
            voxel_size=self.voxel_size,
            device=self.device,
        )
        
        # #region agent log
        if torch.cuda.is_available():
            num_voxels = sparse_feat.feats.shape[0] if hasattr(sparse_feat, 'feats') else 0
            sparse_feat_size = sparse_feat.feats.numel() * 4 / 1024**2 if hasattr(sparse_feat, 'feats') else 0
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After construct_sparse_tensor, before sparse_conv",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_voxels": num_voxels,
                    "vol_dim": vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim,
                    "sparse_feat_size_mb": sparse_feat_size,
                    "valid_coords_size_mb": valid_coords.numel() * 4 / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        feat_3d = self.sparse_conv(sparse_feat)
        
        # #region agent log
        if torch.cuda.is_available():
            # sparse_conv returns torch.Tensor (x.F), not SparseTensor
            if isinstance(feat_3d, torch.Tensor):
                feat_3d_size = feat_3d.numel() * 4 / 1024**2
                feat_3d_shape = list(feat_3d.shape)
            elif hasattr(feat_3d, 'feats'):
                feat_3d_size = feat_3d.feats.numel() * 4 / 1024**2
                feat_3d_shape = list(feat_3d.feats.shape)
            else:
                feat_3d_size = 0
                feat_3d_shape = None
            
            # Calculate expected dense volume size
            vol_dim_list = vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim
            if isinstance(feat_3d, torch.Tensor):
                outdim = feat_3d.shape[-1]
            elif hasattr(feat_3d, 'feats'):
                outdim = feat_3d.feats.shape[-1]
            else:
                outdim = 32  # default
            expected_dense_size_mb = vol_dim_list[0] * vol_dim_list[1] * vol_dim_list[2] * outdim * 4 / 1024**2
            
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After sparse_conv, before sparse_to_dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_size_mb": feat_3d_size,
                    "feat_3d_shape": feat_3d_shape,
                    "feat_3d_type": str(type(feat_3d)),
                    "expected_dense_size_mb": expected_dense_size_mb,
                    "vol_dim": vol_dim_list,
                    "outdim": outdim,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        # 记录调用 sparse_to_dense_volume 前的内存状态作为基线
        memory_before_dense = None
        if torch.cuda.is_available():
            memory_before_dense = torch.cuda.memory_allocated() / 1024**2
        
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Right before sparse_to_dense_volume (critical memory point)",
                {
                    "allocated_mb": memory_before_dense,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        try:
            dense_volume = self.sparse_to_dense_volume(
                sparse_tensor=feat_3d,
                coords=valid_coords,
                vol_dim=vol_dim,
            ).unsqueeze(dim=0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:_build_3d_feature_volume",
                        "OOM ERROR in sparse_to_dense_volume",
                        {
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "error": str(e),
                            "vol_dim": vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
            raise
        
        # #region agent log
        if torch.cuda.is_available():
            vol_dim_list = vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim
            dense_volume_size = dense_volume.numel() * 4 / 1024**2
            expected_dense_size = vol_dim_list[0] * vol_dim_list[1] * vol_dim_list[2] * dense_volume.shape[-1] * 4 / 1024**2
            memory_after_dense = torch.cuda.memory_allocated() / 1024**2
            memory_increase_mb = memory_after_dense - memory_before_dense if memory_before_dense is not None else None
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After sparse_to_dense_volume, before permute",
                {
                    "allocated_mb": memory_after_dense,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "dense_volume_size_mb": dense_volume_size,
                    "dense_volume_shape": list(dense_volume.shape),
                    "expected_dense_size_mb": expected_dense_size,
                    "vol_dim": vol_dim_list,
                    "memory_increase_mb": memory_increase_mb,  # Compare with before sparse_to_dense_volume
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)

        grid_coords_bg = self.get_grid_coords(means_bg, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop_bg = self.interpolate_features(grid_coords_bg, dense_volume)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After interpolate_features for bg",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_bg_size_mb": feat_3d_crop_bg.numel() * 4 / 1024**2,
                    "feat_3d_crop_bg_shape": list(feat_3d_crop_bg.shape),
                },
                hypothesis_id="H5",
            )
        # #endregion

        if node_state_rigid is not None and means_rigid_world_all.shape[0] > 0:
            feat_dim = feat_3d_crop_bg.shape[1]
            feat_3d_crop_rigid = torch.zeros(
                means_rigid_world_all.shape[0],
                feat_dim,
                device=self.device,
            )
            if rigid_in_crop_mask is not None and rigid_in_crop_mask.any():
                means_in_crop = means_rigid_world_all[rigid_in_crop_mask]
                grid_coords_rigid_in_crop = self.get_grid_coords(
                    means_in_crop, self.bbx_min, vol_dim, self.voxel_size
                )
                feat_3d_rigid_in_crop = self.interpolate_features(
                    grid_coords_rigid_in_crop, dense_volume
                )
                feat_3d_crop_rigid[rigid_in_crop_mask] = feat_3d_rigid_in_crop
            if rigid_visible_mask is not None:
                feat_3d_crop_rigid = feat_3d_crop_rigid * rigid_visible_mask[:, None].float()
        else:
            feat_3d_crop_rigid = torch.empty(0, feat_3d_crop_bg.shape[1], device=self.device)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Before deleting dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_rigid_size_mb": feat_3d_crop_rigid.numel() * 4 / 1024**2 if feat_3d_crop_rigid.numel() > 0 else 0,
                },
                hypothesis_id="H5",
            )
        # #endregion

        del dense_volume
        
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After deleting dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        return feat_3d_crop_bg, feat_3d_crop_rigid, rigid_visible_mask, rigid_in_crop_mask

    def _prepare_gaussians_for_source(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        合并静态与动态高斯参数（动态先变换到 source 帧），用于 2D 特征反投影。
        返回合并后的高斯字典以及静态/动态数量。
        """
        num_sh = _num_sh_bases(self.sh_degree)
        means_bg = node_state_bg.means
        quats_bg = node_state_bg.quats
        scales_bg = torch.exp(node_state_bg.scales_log)
        opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
        colors_bg = torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)

        means_rigid_world = torch.empty(0, 3, device=self.device)
        quats_rigid_world = torch.empty(0, 4, device=self.device)
        scales_rigid = torch.empty(0, 3, device=self.device)
        opacities_rigid = torch.empty(0, device=self.device)
        colors_rigid = torch.zeros(0, num_sh, 3, device=self.device)
        if node_state_rigid is not None and node_state_rigid.means.numel() > 0:
            node_state_rigid.cur_frame = source_frame_idx
            means_rigid_world = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, node_state_rigid.quats)
            scales_rigid = torch.exp(node_state_rigid.scales_log)
            opacities_rigid = torch.sigmoid(node_state_rigid.opacity_logit).squeeze(-1)
            colors_rigid = torch.cat([node_state_rigid.sh_dc[:, None, :], node_state_rigid.sh_rest], dim=1)

        gaussians = {
            "means": torch.cat([means_bg, means_rigid_world], dim=0),
            "quats": torch.cat([quats_bg, quats_rigid_world], dim=0),
            "scales": torch.cat([scales_bg, scales_rigid], dim=0),
            "opacities": torch.cat([opacities_bg, opacities_rigid], dim=0),
            "colors": torch.cat([colors_bg, colors_rigid], dim=0),
        }
        return gaussians, means_bg.shape[0], means_rigid_world.shape[0]

    def _prepare_all_gaussians(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        source_frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, int]:
        """
        合并三类点（前景、动态、背景远景）用于 2D 特征计算。
        """
        num_sh = _num_sh_bases(self.sh_degree)

        means_bg = node_state_bg.means
        quats_bg = node_state_bg.quats
        scales_bg = torch.exp(node_state_bg.scales_log)
        opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
        colors_bg = torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)
        num_bg = means_bg.shape[0]

        means_rigid_world = torch.empty(0, 3, device=self.device)
        quats_rigid_world = torch.empty(0, 4, device=self.device)
        scales_rigid = torch.empty(0, 3, device=self.device)
        opacities_rigid = torch.empty(0, device=self.device)
        colors_rigid = torch.zeros(0, num_sh, 3, device=self.device)
        num_rigid = 0
        if node_state_rigid is not None and node_state_rigid.means.numel() > 0:
            node_state_rigid.cur_frame = source_frame_idx
            # 解析帧索引以确保使用正确的帧位姿
            resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, source_frame_idx)
            means_rigid_world = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, node_state_rigid.quats)
            scales_rigid = torch.exp(node_state_rigid.scales_log)
            opacities_rigid = torch.sigmoid(node_state_rigid.opacity_logit).squeeze(-1)
            colors_rigid = torch.cat([node_state_rigid.sh_dc[:, None, :], node_state_rigid.sh_rest], dim=1)
            num_rigid = means_rigid_world.shape[0]

        means_distant = torch.empty(0, 3, device=self.device)
        quats_distant = torch.empty(0, 4, device=self.device)
        scales_distant = torch.empty(0, 3, device=self.device)
        opacities_distant = torch.empty(0, device=self.device)
        colors_distant = torch.zeros(0, num_sh, 3, device=self.device)
        num_distant = 0
        if node_state_distant is not None and node_state_distant.means.numel() > 0:
            means_distant = node_state_distant.means
            quats_distant = node_state_distant.quats
            scales_distant = torch.exp(node_state_distant.scales_log)
            opacities_distant = torch.sigmoid(node_state_distant.opacity_logit).squeeze(-1)
            colors_distant = torch.cat([node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1)
            num_distant = means_distant.shape[0]

        gaussians = {
            "means": torch.cat([means_bg, means_rigid_world, means_distant], dim=0),
            "quats": torch.cat([quats_bg, quats_rigid_world, quats_distant], dim=0),
            "scales": torch.cat([scales_bg, scales_rigid, scales_distant], dim=0),
            "opacities": torch.cat([opacities_bg, opacities_rigid, opacities_distant], dim=0),
            "colors": torch.cat([colors_bg, colors_rigid, colors_distant], dim=0),
        }
        return gaussians, num_bg, num_rigid, num_distant

    def _compute_2d_features(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_views: List,
        source_images: List[torch.Tensor],
        source_frame_idx: int,
        rigid_visible_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        双轮渲染：先渲染 RGB 供 CNN 使用，再流式渲染提取权重并反投影。
        """
        if (
            not self.use_2d_features
            or self.image_feature_extractor is None
            or self.alpha_t_extractor is None
            or self.feature_backprojector is None
        ):
            return None, None
        if source_images is None or len(source_images) == 0 or source_views is None or len(source_views) == 0:
            return None, None

        imgs = [img.to(self.device) for img in source_images if img is not None]
        if len(imgs) == 0:
            return None, None
        sample_img = imgs[0]
        if sample_img.dim() == 3 and sample_img.shape[-1] == 3:
            height, width = sample_img.shape[0], sample_img.shape[1]
        elif sample_img.dim() == 3 and sample_img.shape[0] == 3:
            height, width = sample_img.shape[1], sample_img.shape[2]
        else:
            height, width = sample_img.shape[-2], sample_img.shape[-1]
        image_batch = torch.stack(imgs, dim=0)

        # Step 1: Prepare Gaussians
        gaussians, num_bg, num_rigid = self._prepare_gaussians_for_source(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            source_frame_idx=source_frame_idx,
        )
        
        # Phase 1: Render RGB only (meta discarded immediately)
        rendered_rgbs = self.alpha_t_extractor.render_rgb_only(
            gaussians, source_views, height, width
        )

        # Convert images to [V, H, W, 3] format if needed
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)

        rendered_batch = torch.stack(rendered_rgbs, dim=0)  # [V, H, W, 3]
        del rendered_rgbs

        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        multi_channel_input = torch.cat([image_batch, rendered_batch], dim=-1)  # [V, H, W, 6]
        del rendered_batch, image_batch

        # Phase 2: CNN forward then streaming backprojection
        features_2d = self.image_feature_extractor(multi_channel_input)  # [V, H_feat, W_feat, C]
        del multi_channel_input

        # Important: reuse the same gaussians for both passes to keep RGB/weights aligned.
        feat_2d_all = self.alpha_t_extractor.render_and_backproject_streaming(
            gaussians=gaussians,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=num_bg + num_rigid,
            backprojector=self.feature_backprojector,
        )

        feat_2d_bg = feat_2d_all[:num_bg]
        feat_2d_rigid = feat_2d_all[num_bg:]
        if rigid_visible_mask is not None and feat_2d_rigid.shape[0] == rigid_visible_mask.shape[0]:
            feat_2d_rigid = feat_2d_rigid * rigid_visible_mask.float().unsqueeze(-1)
        return feat_2d_bg, feat_2d_rigid

    def _compute_2d_features_all(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        source_views: List,
        source_images: List[torch.Tensor],
        source_frame_idx: int,
        rigid_visible_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算所有点（前景+动态+背景远景）的 2D 特征。
        """
        if (
            not self.use_2d_features
            or self.image_feature_extractor is None
            or self.alpha_t_extractor is None
            or self.feature_backprojector is None
        ):
            return None, None, None
        if source_images is None or len(source_images) == 0 or source_views is None or len(source_views) == 0:
            return None, None, None

        imgs = [img.to(self.device) for img in source_images if img is not None]
        if len(imgs) == 0:
            return None, None, None
        sample_img = imgs[0]
        if sample_img.dim() == 3 and sample_img.shape[-1] == 3:
            height, width = sample_img.shape[0], sample_img.shape[1]
        elif sample_img.dim() == 3 and sample_img.shape[0] == 3:
            height, width = sample_img.shape[1], sample_img.shape[2]
        else:
            height, width = sample_img.shape[-2], sample_img.shape[-1]
        image_batch = torch.stack(imgs, dim=0)

        gaussians_all, num_bg, num_rigid, num_distant = self._prepare_all_gaussians(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            node_state_distant=node_state_distant,
            source_frame_idx=source_frame_idx,
        )
        total_points = num_bg + num_rigid + num_distant
        if total_points == 0:
            return None, None, None

        rendered_rgbs = self.alpha_t_extractor.render_rgb_only(
            gaussians_all, source_views, height, width
        )

        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)

        rendered_batch = torch.stack(rendered_rgbs, dim=0)
        del rendered_rgbs

        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        multi_channel_input = torch.cat([image_batch, rendered_batch], dim=-1)
        del rendered_batch, image_batch

        features_2d = self.image_feature_extractor(multi_channel_input)
        del multi_channel_input

        feat_2d_all = self.alpha_t_extractor.render_and_backproject_streaming(
            gaussians=gaussians_all,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=total_points,
            backprojector=self.feature_backprojector,
        )

        feat_2d_bg = feat_2d_all[:num_bg] if num_bg > 0 else None
        feat_2d_rigid = feat_2d_all[num_bg:num_bg + num_rigid] if num_rigid > 0 else None
        feat_2d_distant = feat_2d_all[num_bg + num_rigid:] if num_distant > 0 else None

        if feat_2d_rigid is not None and rigid_visible_mask is not None and feat_2d_rigid.shape[0] == rigid_visible_mask.shape[0]:
            feat_2d_rigid = feat_2d_rigid * rigid_visible_mask.float().unsqueeze(-1)

        return feat_2d_bg, feat_2d_rigid, feat_2d_distant

    def _fuse_features(
        self,
        feat_3d: torch.Tensor,
        feat_2d: Optional[torch.Tensor],
        visibility: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        融合 2D/3D 特征与可见性标量。
        """
        if not self.use_2d_features or feat_2d is None or self.feature_fusion is None:
            return feat_3d
        if visibility is None:
            visibility = torch.ones(feat_3d.shape[0], device=feat_3d.device)
        fused = self.feature_fusion.fuse(feat_3d, feat_2d, visibility)
        return fused

    def _mask_rigid_offsets(
        self, offsets: Dict[str, torch.Tensor], visible_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        使用可见性掩码屏蔽动态物体的偏移量。
        
        Args:
            offsets: 偏移量字典
            visible_mask: 可见性掩码，形状 [N_rigid]，bool 类型，可选
            
        Returns:
            屏蔽后的偏移量字典
        
        对于不可见的点，将偏移量置零（位置、尺度、不透明度、SH）或设为单位四元数（旋转）。
        """
        if visible_mask is None or visible_mask.numel() == 0:
            return offsets
        mask = visible_mask.to(offsets["offset_pos"].device)
        mask_vec = mask.unsqueeze(-1).float()
        offset_quat = offsets["offset_quat"]
        identity_quat = torch.zeros_like(offset_quat)
        identity_quat[..., 0] = 1.0
        masked_offsets = {
            "offset_pos": offsets["offset_pos"] * mask_vec,
            "offset_scales": offsets["offset_scales"] * mask_vec,
            "offset_quat": torch.where(mask.unsqueeze(-1), offset_quat, identity_quat),
            "offset_opacity": offsets["offset_opacity"] * mask_vec,
            "offset_sh": offsets["offset_sh"] * mask_vec,
        }
        return masked_offsets

    def get_grid_coords(
        self, position_w: torch.Tensor, bbx_min: torch.Tensor, vol_dim, voxel_size: float
    ) -> torch.Tensor:
        """
        将世界坐标转换为体积网格的归一化坐标（用于 grid_sample）。
        
        Args:
            position_w: 世界坐标位置，形状 [N, 3]
            bbx_min: 边界框最小值，形状 [3]
            vol_dim: 体积维度，[D, H, W] 格式（可以是 list、tuple 或 Tensor）
            voxel_size: 体素大小（米）
            
        Returns:
            归一化网格坐标，形状 [N, 3]，格式 [x_norm, y_norm, z_norm]，值域 [-1, 1]
        
        处理流程：
        1. 将坐标相对于边界框原点：pts = position_w - bbx_min
        2. 转换为体素索引（浮点数）：index = pts / voxel_size
        3. 归一化到 [-1, 1] 范围：norm = 2.0 * (index / (vol_dim - 1)) - 1.0
        4. 堆叠为 [x_norm, y_norm, z_norm] 格式（grid_sample 要求）
        
        注意：
        - grid_sample (5D) 期望坐标顺序为 [x, y, z]，对应 [W, H, D] 维度
        - 由于 dense_volume 是 [1, C, Z, Y, X] = [B, C, D, H, W]，其中 D=Z, H=Y, W=X
        - 因此 grid 坐标必须是 [x_norm, y_norm, z_norm] 对应 [W, H, D]
        - 使用 align_corners=True，所以 index 0 映射到 -1.0，index (N-1) 映射到 1.0
        """
        # Clamp positions to bbox range to match construct_sparse_tensor behavior
        # This ensures coordinates are within the volume bounds
        # Use self.bbx_max directly instead of recalculating from vol_dim
        bbx_max = self.bbx_max.to(position_w.device)
        position_w_clamped = torch.clamp(position_w, min=bbx_min, max=bbx_max)
        
        pts = position_w_clamped - bbx_min.to(position_w.device)
        x_index = pts[..., 0] / voxel_size
        y_index = pts[..., 1] / voxel_size
        z_index = pts[..., 2] / voxel_size
        
        # Convert vol_dim to torch.Tensor if it's a list, tuple, or numpy array
        # construct_sparse_tensor may return Python list from nerfstudio implementation
        if isinstance(vol_dim, (list, tuple)):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        elif not isinstance(vol_dim, torch.Tensor):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        else:
            vol_dim = vol_dim.to(position_w.device).float()
        
        # vol_dim is [X, Y, Z] from construct_sparse_tensor (world coordinates)
        # sparse_to_dense_volume creates dense volume as [X, Y, Z, C]
        # After unsqueeze: [1, X, Y, Z, C]
        # After permute(0, 4, 3, 2, 1): [1, C, Z, Y, X] = [1, C, D, H, W]
        # where D=Z, H=Y, W=X
        # 
        # PyTorch grid_sample (5D) for input [B, C, D, H, W] expects grid coordinates
        # in the format [x, y, z] corresponding to [W, H, D] dimensions.
        # This is the standard convention: the last dimension of grid is [x, y, z]
        # which maps to [width, height, depth] of the input volume.
        #
        # Therefore, we must return [x_norm, y_norm, z_norm] to match [W, H, D].
        # For align_corners=True: index 0 maps to -1.0, index (N-1) maps to 1.0
        # Therefore, we use (vol_dim - 1) as denominator to ensure correct boundary mapping
        den_x = torch.clamp(vol_dim[0] - 1.0, min=1.0)
        den_y = torch.clamp(vol_dim[1] - 1.0, min=1.0)
        den_z = torch.clamp(vol_dim[2] - 1.0, min=1.0)
        x_norm = 2.0 * (x_index / den_x) - 1.0  # X -> W
        y_norm = 2.0 * (y_index / den_y) - 1.0  # Y -> H
        z_norm = 2.0 * (z_index / den_z) - 1.0  # Z -> D
        # grid_sample (5D) expects coordinates in [x, y, z] order for [B, C, D, H, W] input
        # This corresponds to [W, H, D] = [X, Y, Z]
        grid_coords = torch.stack([x_norm, y_norm, z_norm], dim=-1)
        
        return grid_coords

    def interpolate_features(self, grid_coords: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
        """
        从 3D 特征体积中插值提取每个点的特征。
        
        Args:
            grid_coords: 归一化网格坐标，形状 [N, 3]，格式 [x_norm, y_norm, z_norm]
            feature_volume: 特征体积，形状 [1, C, D, H, W]（经过 permute 后，其中 D=Z, H=Y, W=X）
            
        Returns:
            每个点的特征，形状 [N, C]
        
        使用三线性插值（grid_sample 在 3D 中）从体积中提取特征。
        grid_sample 期望输入格式为 [B, C, D, H, W]，坐标格式为 [B, D_out, H_out, W_out, 3]。
        我们扩展 grid_coords 为 [1, 1, 1, N, 3] 以匹配要求。
        """
        grid_coords_expanded = grid_coords[None, None, None, ...]
        feature = torch.nn.functional.grid_sample(
            feature_volume,
            grid_coords_expanded,
            mode="bilinear",  # 在3D中实际是三线性插值
            align_corners=True,
            padding_mode="zeros",
        )
        return feature[0, :, 0, 0, :].T  # [1, C, 1, 1, N] → [C, N] → [N, C]

    def _predict_offsets(self, feat_3d_crop: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从 3D 特征预测 Gaussian 参数的偏移量。
        
        Args:
            feat_3d_crop: 3D 特征，形状 [N, outdim]（默认 outdim=32）
            
        Returns:
            偏移量字典，包含：
                - "offset_pos": 位置偏移，形状 [N, 3]，范围 [-offset_max, offset_max]
                - "offset_scales": 尺度对数偏移，形状 [N, 3]，范围 [-scale_max, scale_max]
                - "offset_quat": 四元数偏移，形状 [N, 4]，wxyz 格式（从轴角转换）
                - "offset_opacity": 不透明度对数偏移，形状 [N, 1]，范围 [-opacity_max, opacity_max]
                - "offset_sh": SH系数偏移，形状 [N, 3*num_sh]，包含DC和rest分量
        
        处理流程：
        1. 位置偏移：mlp_offset_pos → tanh → offset_max 缩放
        2. 尺度与旋转：mlp_conv → 分离尺度和轴角 → 分别 tanh 限制 → 轴角转四元数
        3. 不透明度偏移：mlp_opacity → tanh → opacity_max 缩放
        4. SH系数偏移：gaussion_decoder → 分离DC和rest → 分别 tanh 限制 → 合并
        
        注意：静态和动态使用相同的 MLP 网络预测偏移量。
        """
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_predict_offsets",
                "Start predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_size_mb": feat_3d_crop.numel() * 4 / 1024**2,
                    "feat_3d_crop_shape": list(feat_3d_crop.shape),
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        # Position offset with tanh clamping
        offset_pos = self.offset_max * torch.tanh(self.mlp_offset_pos(feat_3d_crop))
        
        # Scale and rotation offsets
        scales_and_omega = self.mlp_conv(feat_3d_crop)
        offset_scales_raw, offset_omega_raw = scales_and_omega.split([3, 3], dim=-1)
        offset_scales = self.scale_max * torch.tanh(offset_scales_raw)
        offset_omega = self.omega_max * torch.tanh(offset_omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        
        # Opacity offset with tanh clamping
        offset_opacity = self.opacity_max * torch.tanh(self.mlp_opacity(feat_3d_crop))
        
        # SH offsets with separate DC and rest
        sh_raw = self.gaussion_decoder(feat_3d_crop)
        sh_dc_raw = sh_raw[:, :3]
        sh_rest_raw = sh_raw[:, 3:]
        offset_sh_dc = self.sh_dc_max * torch.tanh(sh_dc_raw)
        offset_sh_rest = self.sh_rest_max * torch.tanh(sh_rest_raw)
        offset_sh = torch.cat([offset_sh_dc, offset_sh_rest], dim=-1)
        
        # #region agent log
        if torch.cuda.is_available():
            total_offset_size = (
                offset_pos.numel() * 4 + offset_scales.numel() * 4 + offset_quat.numel() * 4 +
                offset_opacity.numel() * 4 + offset_sh.numel() * 4
            ) / 1024**2
            _debug_log(
                "streetforward.py:_predict_offsets",
                "After predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "total_offset_size_mb": total_offset_size,
                    "offset_shapes": {
                        "offset_pos": list(offset_pos.shape),
                        "offset_scales": list(offset_scales.shape),
                        "offset_quat": list(offset_quat.shape),
                        "offset_opacity": list(offset_opacity.shape),
                        "offset_sh": list(offset_sh.shape),
                    },
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        return {
            "offset_pos": offset_pos,
            "offset_scales": offset_scales,
            "offset_quat": offset_quat,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
        }

    def _render_params_from_offsets(
        self, node_state: NodeState, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        从 NodeState 和偏移量计算渲染参数。
        
        Args:
            node_state: NodeState（Background 或 RigidNodes），所有参数都是分离的
            offsets: 偏移量字典（可微）
            
        Returns:
            渲染参数字典，包含：
                - "means_r": 渲染用的位置，形状 [N, 3]（可微，未clamp）
                - "scales_log_r": 渲染用的尺度对数，形状 [N, 3]（可微）
                - "scales_r": 渲染用的尺度，形状 [N, 3]（exp(scales_log_r)）
                - "quats_r": 渲染用的四元数，形状 [N, 4]（归一化，可微）
                - "opacity_logit_r": 渲染用的不透明度对数，形状 [N, 1]（可微）
                - "opacities_r": 渲染用的不透明度，形状 [N]（sigmoid(opacity_logit_r)）
                - "sh_dc_r": 渲染用的SH DC分量，形状 [N, 3]（可微）
                - "sh_rest_r": 渲染用的SH高阶分量，形状 [N, num_sh-1, 3]（可微）
                - "colors_r": 完整的SH系数，形状 [N, num_sh, 3]（用于渲染）
        
        关键点：
        - 应用步长因子（eta）控制偏移量幅度
        - means_r 不在此处进行 clamp，以保持梯度流
        - 使用四元数乘法组合旋转
        - 静态背景的渲染参数是世界坐标，动态物体的是局部坐标
        """
        num_points = node_state.means.shape[0]
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)

        # Apply offsets with step size factors (eta)
        # Note: means_r is not clamped here to preserve gradient flow
        means_r = node_state.means + self.eta_means * offsets["offset_pos"]
        scales_log_r = node_state.scales_log + self.eta_scales * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state.opacity_logit + self.eta_opacity * offsets["offset_opacity"]
        sh_dc_r = node_state.sh_dc + self.eta_sh_dc * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state.sh_rest + self.eta_sh_rest * sh_rest_offset

        scales_r = torch.exp(scales_log_r)
        opacities_r = torch.sigmoid(opacity_logit_r).squeeze(-1)
        colors_r = torch.cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)

        return {
            "means_r": means_r,
            "scales_log_r": scales_log_r,
            "quats_r": quats_r,
            "opacity_logit_r": opacity_logit_r,
            "sh_dc_r": sh_dc_r,
            "sh_rest_r": sh_rest_r,
            "scales_r": scales_r,
            "opacities_r": opacities_r,
            "colors_r": colors_r,
        }

    def _create_proxy_params(self, render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        创建代理参数，用于多视角梯度累积。
        
        Args:
            render_params: 渲染参数字典（可微）
            
        Returns:
            代理参数字典，包含：
                - "means_p": 代理位置（分离但可微）
                - "scales_p": 代理尺度（分离但可微）
                - "quats_p": 代理四元数（分离但可微）
                - "opacities_p": 代理不透明度（分离但可微）
                - "colors_p": 代理颜色（分离但可微）
        
        操作：proxy = render_param.detach().requires_grad_(True)
        
        关键点：
        - 代理参数从渲染参数中分离（detach），但重新启用梯度
        - 这样可以在多个视角上累积梯度，然后一次性反向传播到渲染参数
        - 所有 target 帧共享同一组代理参数
        """
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Creating proxy params",
            {
                "num_points": render_params["means_r"].shape[0],
                "requires_grad_before": render_params["means_r"].requires_grad,
            },
            hypothesis_id="H1",
        )
        # #endregion
        proxies = {
            "means_p": render_params["means_r"].detach().requires_grad_(True),
            "scales_p": render_params["scales_r"].detach().requires_grad_(True),
            "quats_p": render_params["quats_r"].detach().requires_grad_(True),
            "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
            "colors_p": render_params["colors_r"].detach().requires_grad_(True),
        }
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Proxy params created",
            {
                "requires_grad_after": proxies["means_p"].requires_grad,
                "is_leaf": proxies["means_p"].is_leaf,
                "grad_fn": str(proxies["means_p"].grad_fn),
            },
            hypothesis_id="H1",
        )
        # #endregion
        return proxies

    def _merge_all_params(
        self,
        proxies_bg: Dict[str, torch.Tensor],
        proxies_rigid: Optional[Dict[str, torch.Tensor]],
        proxies_distant: Optional[Dict[str, torch.Tensor]],
        means_rigid_world: torch.Tensor,
        quats_rigid_world: torch.Tensor,
        opacities_rigid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        合并前景、动态和背景远景的渲染参数。
        """
        means_list = [proxies_bg["means_p"]]
        quats_list = [proxies_bg["quats_p"]]
        scales_list = [proxies_bg["scales_p"]]
        opacities_list = [proxies_bg["opacities_p"]]
        colors_list = [proxies_bg["colors_p"]]

        if proxies_rigid is not None and means_rigid_world.numel() > 0:
            means_list.append(means_rigid_world)
            quats_list.append(quats_rigid_world)
            scales_list.append(proxies_rigid["scales_p"])
            opacities_list.append(opacities_rigid if opacities_rigid is not None else proxies_rigid["opacities_p"])
            colors_list.append(proxies_rigid["colors_p"])

        if proxies_distant is not None:
            means_list.append(proxies_distant["means_p"])
            quats_list.append(proxies_distant["quats_p"])
            scales_list.append(proxies_distant["scales_p"])
            opacities_list.append(proxies_distant["opacities_p"])
            colors_list.append(proxies_distant["colors_p"])

        return (
            torch.cat(means_list, dim=0),
            torch.cat(quats_list, dim=0),
            torch.cat(scales_list, dim=0),
            torch.cat(opacities_list, dim=0),
            torch.cat(colors_list, dim=0),
        )

    def compute_loss(self, pred_rgb: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
        """
        计算 L2 损失（均方误差）。
        
        Args:
            pred_rgb: 预测的RGB图像，形状 [H, W, 3]
            gt_image: 真实图像，形状 [H, W, 3]
            
        Returns:
            标量损失值：mean((pred_rgb - gt_image)²)
        """
        return torch.mean((pred_rgb - gt_image) ** 2)

    def _compute_render_params(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        """
        共享的前向传播：从节点状态计算渲染参数。
        
        Args:
            node_state: NodeState（Background 或 RigidNodes）
            
        Returns:
            渲染参数字典
        
        这是评估时使用的简化流程：
        1. 构建 3D 特征体积（只使用单个 NodeState）
        2. 预测偏移量
        3. 计算渲染参数
        
        注意：此方法不处理动态物体的坐标变换，适用于静态背景或已变换到目标帧的动态物体。
        """
        means_s = node_state.means
        anchor_rgb = _sh_to_rgb(node_state.sh_dc)

        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means_s.clone(),
            feats=anchor_rgb,
            Bbx_max=self.bbx_max,
            Bbx_min=self.bbx_min,
            voxel_size=self.voxel_size,
            device=self.device,
        )
        feat_3d = self.sparse_conv(sparse_feat)
        dense_volume = self.sparse_to_dense_volume(
            sparse_tensor=feat_3d,
            coords=valid_coords,
            vol_dim=vol_dim,
        ).unsqueeze(dim=0)
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
        grid_coords = self.get_grid_coords(means_s, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop = self.interpolate_features(grid_coords, dense_volume)
        del dense_volume

        offsets = self._predict_offsets(feat_3d_crop)
        render_params = self._render_params_from_offsets(node_state, offsets)
        return render_params

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染单个视角，返回 RGB 图像和 alpha 通道。
        
        Args:
            render_params: 渲染参数字典，可以是代理参数（"means_p"等）或渲染参数（"means_r"等）
            view: 相机视角对象，需有 camtoworlds 和 Ks/K 属性
            height: 图像高度
            width: 图像宽度
            
        Returns:
            (rgb, acc) 元组：
                - rgb: RGB 图像，形状 [H, W, 3]
                - acc: 累积不透明度，形状 [H, W]
        
        使用 gsplat 渲染器进行高斯点渲染。
        """
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = get_viewmat(c2w)
        k_mat = None
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=self.device).unsqueeze(0)
        
        # Ensure Ks is [1,3,3] format
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)

        means_key = "means_p" if "means_p" in render_params else "means_r"
        scales_key = "scales_p" if "scales_p" in render_params else "scales_r"
        quats_key = "quats_p" if "quats_p" in render_params else "quats_r"
        opacities_key = "opacities_p" if "opacities_p" in render_params else "opacities_r"
        colors_key = "colors_p" if "colors_p" in render_params else "colors_r"

        render, alpha, _ = self.renderer(
            means=render_params[means_key],
            quats=render_params[quats_key],
            scales=render_params[scales_key],
            opacities=render_params[opacities_key],
            colors=render_params[colors_key],
            viewmats=viewmat,
            Ks=k_mat,
            width=width,
            height=height,
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )

        rgb = render[:, ..., :3].squeeze(0)
        acc = alpha.squeeze(0)
        return rgb, acc

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
            feat_bg, feat_rigid, rigid_visible_mask, rigid_in_crop_mask = self._build_3d_feature_volume(
                node_state_bg=node_state_bg,
                node_state_rigid=node_state_rigid,
                source_frame_idx=source_frame_idx,
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
                resolved_frame_idx = None
                if node_state_rigid is not None:
                    node_state_rigid.cur_frame = target_frame_idx
                    resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, target_frame_idx)
                if proxies_rigid is not None and node_state_rigid is not None:
                    # #region agent log
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before rigid transform for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "target_frame_idx": target_frame_idx,
                            "proxies_rigid_means_requires_grad": proxies_rigid["means_p"].requires_grad,
                            "proxies_rigid_means_grad": proxies_rigid["means_p"].grad is not None,
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                    means_rigid_world = self._transform_rigid_to_world(node_state_rigid, proxies_rigid["means_p"])
                    quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, proxies_rigid["quats_p"])
                    # #region agent log
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After rigid transform for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "means_rigid_world_requires_grad": means_rigid_world.requires_grad,
                            "means_rigid_world_grad_fn": str(means_rigid_world.grad_fn),
                            "quats_rigid_world_requires_grad": quats_rigid_world.requires_grad,
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                    if resolved_frame_idx is not None:
                        visibility = node_state_rigid.instances_fv[resolved_frame_idx]
                        valid_mask = visibility[node_state_rigid.point_ids[..., 0]].float()
                        opacities_rigid = proxies_rigid["opacities_p"] * valid_mask
                    else:
                        opacities_rigid = proxies_rigid["opacities_p"]
                else:
                    means_rigid_world = torch.empty(0, 3, device=self.device)
                    quats_rigid_world = torch.empty(0, 4, device=self.device)
                    opacities_rigid = None
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

    @torch.no_grad()
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

    def save_checkpoint(
        self,
        step: Optional[int] = None,
        is_final: bool = False,
        checkpoint_dir: Optional[str] = None,
    ) -> str:
        """
        持久化模型/优化器和分离的节点状态。
        
        Args:
            step: 可选的训练步数（默认为 self.global_step）
            is_final: 如果为 True，总是写入 checkpoint_final.pth
            checkpoint_dir: 覆盖输出目录
            
        Returns:
            检查点文件路径
        
        保存内容：
        - 模型状态（sparse_conv、所有 MLP 头）
        - 优化器状态
        - 所有 NodeStateBackground（静态背景）
        - 所有 NodeStateRigid（动态物体）
        - 配置（如果可序列化）
        """
        step_val = int(step if step is not None else self.global_step)
        ckpt_dir = (
            checkpoint_dir
            or self.checkpoint_dir
            or (os.path.join(self.config.log_dir, "checkpoints") if hasattr(self.config, "log_dir") else None)
            or "./checkpoints"
        )
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        filename = "checkpoint_final.pth" if is_final else f"checkpoint_step_{step_val:06d}.pth"
        checkpoint_path = os.path.join(ckpt_dir, filename)

        model_state_dict = {
            "sparse_conv": self.sparse_conv.state_dict(),
            "mlp_offset_pos": self.mlp_offset_pos.state_dict(),
            "mlp_conv": self.mlp_conv.state_dict(),
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "gaussion_decoder": self.gaussion_decoder.state_dict(),
        }
        if self.image_feature_extractor is not None:
            model_state_dict["image_feature_extractor"] = self.image_feature_extractor.state_dict()

        nodes_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states.items()
        }
        rigid_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_rigid_to_dict(state)
            for (scene, segment), state in self.node_states_rigid.items()
            if state is not None
        }
        distant_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states_distant.items()
            if state is not None
        }

        checkpoint = {
            "step": step_val,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "node_states": nodes_state_dict,
            "node_states_rigid": rigid_state_dict,
            "node_states_distant": distant_state_dict,
        }
        try:
            checkpoint["config"] = OmegaConf.to_container(self.config, resolve=False)
        except Exception:
            logger.debug("Config not serialized into checkpoint (non-fatal).")

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> int:
        """
        恢复模型/优化器和节点状态。
        
        Args:
            checkpoint_path: .pth 检查点文件路径
            load_optimizer: 如果可用，加载优化器状态
            strict: 权重加载的严格性
            
        Returns:
            恢复的 global_step
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        self.sparse_conv.load_state_dict(model_state["sparse_conv"], strict=strict)
        self.mlp_offset_pos.load_state_dict(model_state["mlp_offset_pos"], strict=strict)
        self.mlp_conv.load_state_dict(model_state["mlp_conv"], strict=strict)
        self.mlp_opacity.load_state_dict(model_state["mlp_opacity"], strict=strict)
        self.gaussion_decoder.load_state_dict(model_state["gaussion_decoder"], strict=strict)
        if "image_feature_extractor" in model_state and self.image_feature_extractor is not None:
            self.image_feature_extractor.load_state_dict(model_state["image_feature_extractor"], strict=strict)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        nodes_state_dict = checkpoint.get("node_states") or checkpoint.get("nodes_state_dict")
        if nodes_state_dict is not None:
            restored_nodes: Dict[Tuple[int, int], NodeState] = {}
            for key, state in nodes_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_nodes[(scene_id, segment_id)] = self._node_state_from_dict(state)
            if restored_nodes:
                self.node_states = restored_nodes
                self.node_states_bg = self.node_states

        rigid_state_dict = checkpoint.get("node_states_rigid")
        if rigid_state_dict is not None:
            restored_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
            for key, state in rigid_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_rigid[(scene_id, segment_id)] = self._node_state_rigid_from_dict(state)
            if restored_rigid:
                self.node_states_rigid = restored_rigid

        distant_state_dict = checkpoint.get("node_states_distant")
        if distant_state_dict is not None:
            restored_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]] = {}
            for key, state in distant_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_distant[(scene_id, segment_id)] = self._node_state_distant_from_dict(state)
            if restored_distant:
                self.node_states_distant = restored_distant

        self.global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
        logger.info(f"Checkpoint loaded from {checkpoint_path} (step={self.global_step})")
        return self.global_step

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
