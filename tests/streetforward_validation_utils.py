import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


@contextmanager
def _stubbed_modules():
    """Temporarily inject lightweight stubs for heavy deps, then restore."""
    import types

    targets = ["models.evol_splat", "gsplat", "gsplat.rendering"]
    backup = {name: sys.modules.get(name) for name in targets}

    dummy_evol = types.ModuleType("models.evol_splat")
    dummy_evol.SparseCostRegNet = None
    dummy_evol.construct_sparse_tensor = None
    dummy_evol.sparse_to_dense_volume = None

    dummy_gsplat = types.ModuleType("gsplat")
    dummy_gsplat_render = types.ModuleType("gsplat.rendering")

    def _stub_rasterization(*args, **kwargs):
        raise ImportError("gsplat stubbed for tests")

    dummy_gsplat_render.rasterization = _stub_rasterization
    dummy_gsplat.rendering = dummy_gsplat_render

    sys.modules["models.evol_splat"] = dummy_evol
    sys.modules["gsplat"] = dummy_gsplat
    sys.modules["gsplat.rendering"] = dummy_gsplat_render
    try:
        yield
    finally:
        for name, mod in backup.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _load_streetforward_module():
    """Load models/trainers/streetforward.py without importing trainers.__init__."""
    module_name = "streetforward_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    import importlib.util

    module_path = Path(__file__).parent.parent / "models" / "trainers" / "streetforward.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with _stubbed_modules():
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return mod


_sf = _load_streetforward_module()
NodeStateBackground = _sf.NodeStateBackground
NodeStateRigid = _sf.NodeStateRigid
StreetForwardTrainer = _sf.StreetForwardTrainer
_axis_angle_to_quat = _sf._axis_angle_to_quat
_num_sh_bases = _sf._num_sh_bases
_normalize_quat = _sf._normalize_quat
_quat_multiply = _sf._quat_multiply


# ---------- Seed helper ----------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Stub sparse structures ----------
@dataclass
class SimpleSparseTensor:
    feats: torch.Tensor
    coords: torch.Tensor

    @property
    def shape(self):
        return self.feats.shape

    def to(self, device: torch.device) -> "SimpleSparseTensor":
        self.feats = self.feats.to(device)
        self.coords = self.coords.to(device)
        return self


def stub_construct_sparse_tensor(
    raw_coords,
    feats,
    Bbx_max,
    Bbx_min,
    voxel_size: float,
    device: torch.device,
):
    raw_coords_t = torch.as_tensor(raw_coords, device=device, dtype=torch.float32)
    feats_t = torch.as_tensor(feats, device=device, dtype=torch.float32)
    bbx_min_t = torch.as_tensor(Bbx_min, device=device, dtype=torch.float32)
    bbx_max_t = torch.as_tensor(Bbx_max, device=device, dtype=torch.float32)

    extent = (bbx_max_t - bbx_min_t).clamp(min=1e-3)
    vol_dim = torch.ceil(extent / voxel_size).long().clamp(min=2)

    coords_float = (raw_coords_t - bbx_min_t) / voxel_size
    coords_int = torch.floor(coords_float).long()
    coords_int = torch.clamp(coords_int, min=0)
    sparse = SimpleSparseTensor(feats=feats_t, coords=coords_int)
    valid_coords = coords_int
    return sparse, vol_dim, valid_coords


class StubSparseConv(nn.Module):
    def __init__(self, d_in: int = 3, d_out: int = 8):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, sparse_feat: SimpleSparseTensor) -> torch.Tensor:
        feats = sparse_feat.feats if hasattr(sparse_feat, "feats") else sparse_feat
        return self.linear(feats)


def stub_sparse_to_dense_volume(
    sparse_tensor,
    coords: torch.Tensor,
    vol_dim,
    default_val: float = 0.0,
):
    feats = sparse_tensor.feats if hasattr(sparse_tensor, "feats") else sparse_tensor
    vol_dim_list = vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else list(vol_dim)
    x, y, z = [int(max(2, d)) for d in vol_dim_list]
    c = feats.shape[-1]
    dense = torch.full((x, y, z, c), default_val, device=feats.device, dtype=feats.dtype)
    if coords.numel() > 0:
        coords_int = coords.long()
        coords_int[:, 0] = coords_int[:, 0].clamp(0, x - 1)
        coords_int[:, 1] = coords_int[:, 1].clamp(0, y - 1)
        coords_int[:, 2] = coords_int[:, 2].clamp(0, z - 1)
        dense[coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = feats
    return dense


# ---------- Stub renderer ----------
def stub_renderer(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
    packed: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    render_mode: str = "RGB",
    sh_degree: int = 1,
    sparse_grad: bool = False,
    absgrad: bool = True,
    rasterize_mode: str = "classic",
):
    device = means.device
    batch = viewmats.shape[0]
    color_term = colors.mean() if colors.numel() > 0 else torch.tensor(0.0, device=device)
    mean_term = means.mean() if means.numel() > 0 else torch.tensor(0.0, device=device)
    scale_term = scales.mean() if scales.numel() > 0 else torch.tensor(0.0, device=device)
    opacity_term = opacities.mean() if opacities.numel() > 0 else torch.tensor(0.0, device=device)

    fused = torch.tanh(mean_term + 0.1 * scale_term + 0.05 * color_term)
    rgb = fused + 0.1 * opacity_term
    render = torch.ones(batch, height, width, 4, device=device, dtype=means.dtype) * rgb
    alpha = torch.sigmoid(opacity_term).expand(batch, height, width)
    return render, alpha, {}


# ---------- Minimal data builders ----------
@dataclass
class DummyView:
    camtoworlds: torch.Tensor
    Ks: torch.Tensor


def make_dummy_view(device: torch.device) -> DummyView:
    c2w = torch.eye(4, device=device).unsqueeze(0)
    k = torch.eye(3, device=device).unsqueeze(0)
    return DummyView(c2w, k)


def build_minimal_batch(device: torch.device) -> Dict:
    background = np.array(
        [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            [0.2, 0.0, 0.0, 0.6, 0.4, 0.5],
            [0.0, 0.2, 0.0, 0.4, 0.6, 0.5],
            [0.0, 0.0, 0.2, 0.5, 0.4, 0.6],
        ],
        dtype=np.float32,
    )
    gt_img = torch.zeros(8, 8, 3, device=device)
    view = make_dummy_view(device)
    return {
        "scene_id": 0,
        "segment_id": 0,
        "source_frame_idx": 0,
        "pointcloud": {"background": background},
        "targets": [
            {
                "frame_idx": 0,
                "view": view,
                "gt_image": gt_img,
            }
        ],
    }


# ---------- Trainer factory ----------
def build_stub_trainer(device: torch.device = torch.device("cpu")) -> StreetForwardTrainer:
    cfg = OmegaConf.create(
        {
            "model": {
                "sh_degree": 1,
                "voxel_size": 0.5,
                "bbx_min": [-1.0, -1.0, -1.0],
                "bbx_max": [1.0, 1.0, 1.0],
                "input_aabb": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                "max_iterations": 1,
                "sparseConv_outdim": 8,
                "use_2d_features": False,
            },
            "optimizer": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.0},
            "training": {"tensorboard": {"enabled": False}},
            "log_images": False,
        }
    )

    trainer = StreetForwardTrainer(
        config=cfg,
        device=device,
        renderer=stub_renderer,
        sparse_conv=StubSparseConv(d_in=3, d_out=cfg.model.sparseConv_outdim),
        construct_sparse_tensor_fn=stub_construct_sparse_tensor,
        sparse_to_dense_volume_fn=stub_sparse_to_dense_volume,
    )

    # Avoid dependency on sklearn in tests
    trainer._compute_initial_scales = lambda means: torch.zeros_like(means)
    return trainer


# ---------- Stats helpers ----------
def _basic_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "l2norm": float(torch.linalg.norm(t).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
        "p50": float(torch.quantile(t, 0.5).item()),
        "p90": float(torch.quantile(t, 0.9).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
    }


def collect_tensor_stats(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for name, tensor in tensors.items():
        if tensor is None or tensor.numel() == 0:
            continue
        stats[name] = _basic_stats(tensor)
    return stats


def collect_grad_norms(
    model: nn.Module,
    patterns: Sequence[str],
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    grad_norms: Dict[str, float] = {}
    matched = 0
    none_grad = 0
    zero_grad = 0
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in patterns):
            continue
        matched += 1
        if param.grad is None:
            none_grad += 1
            grad_norms[name] = 0.0
            continue
        norm = float(param.grad.norm().item())
        grad_norms[name] = norm
        if norm <= eps:
            zero_grad += 1
    counts = {
        "matched_param_count": matched,
        "none_grad_count": none_grad,
        "zero_grad_count": zero_grad,
    }
    return grad_norms, counts


# ---------- Geometry helpers for tests ----------
def make_test_rigid_state(device: torch.device) -> NodeStateRigid:
    means = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    scales_log = torch.zeros(1, 3, device=device)
    quats_local = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opacity_logit = torch.zeros(1, 1, device=device)
    sh_dc = torch.zeros(1, 3, device=device)
    sh_rest = torch.zeros(1, _num_sh_bases(1) - 1, 3, device=device)
    point_ids = torch.tensor([[0]], device=device, dtype=torch.long)

    omega = torch.tensor([[0.0, 0.0, math.pi / 2]], device=device)
    quat_frame = _axis_angle_to_quat(omega)[0]
    instances_quats = quat_frame.view(1, 1, 4)
    instances_trans = torch.tensor([[[1.0, 2.0, 3.0]]], device=device)
    instances_fv = torch.ones(1, 1, device=device, dtype=torch.bool)
    return NodeStateRigid(
        means=means,
        scales_log=scales_log,
        quats=quats_local,
        opacity_logit=opacity_logit,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        point_ids=point_ids,
        instances_quats=instances_quats,
        instances_trans=instances_trans,
        instances_fv=instances_fv,
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )
