from pathlib import Path
import sys
import importlib.util
from types import ModuleType
import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub heavy deps before loading streetforward.py
gsplat_mod = ModuleType("gsplat")
gsplat_render_mod = ModuleType("gsplat.rendering")

def _dummy_rasterization(**kwargs):
    # Mimic gsplat signature loosely; return placeholders
    return torch.empty(0), None, {}

gsplat_render_mod.rasterization = _dummy_rasterization
gsplat_mod.rendering = gsplat_render_mod
sys.modules.setdefault("gsplat", gsplat_mod)
sys.modules.setdefault("gsplat.rendering", gsplat_render_mod)

evol_mod = ModuleType("models.evol_splat")

class _DummySparseCostRegNet(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, sparse_tensor):
        feats = sparse_tensor.feats if hasattr(sparse_tensor, "feats") else sparse_tensor
        return self.linear(feats)

def _dummy_construct_sparse_tensor(raw_coords, feats, Bbx_max, Bbx_min, voxel_size, device):
    return feats, torch.tensor([1, 1, 1]), raw_coords

def _dummy_sparse_to_dense_volume(sparse_tensor, coords, vol_dim):
    return torch.zeros(int(vol_dim[0]), int(vol_dim[1]), int(vol_dim[2]), sparse_tensor.shape[-1])

evol_mod.SparseCostRegNet = _DummySparseCostRegNet
evol_mod.construct_sparse_tensor = _dummy_construct_sparse_tensor
evol_mod.sparse_to_dense_volume = _dummy_sparse_to_dense_volume
sys.modules.setdefault("models.evol_splat", evol_mod)

# Load streetforward.py directly to avoid importing heavy trainer __init__ side effects
_streetforward_spec = importlib.util.spec_from_file_location(
    "models.trainers.streetforward", PROJECT_ROOT / "models" / "trainers" / "streetforward.py"
)
assert _streetforward_spec and _streetforward_spec.loader
_streetforward_module = importlib.util.module_from_spec(_streetforward_spec)
sys.modules.setdefault("models.trainers", ModuleType("models.trainers"))
sys.modules["models.trainers.streetforward"] = _streetforward_module
_streetforward_spec.loader.exec_module(_streetforward_module)

StreetForwardTrainer = _streetforward_module.StreetForwardTrainer
NodeStateBackground = _streetforward_module.NodeStateBackground
NodeStateRigid = _streetforward_module.NodeStateRigid
NodeStateDistant = _streetforward_module.NodeStateDistant

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class _DummySparseTensor:
    def __init__(self, coords: torch.Tensor, feats: torch.Tensor) -> None:
        self.coords = coords
        self.feats = feats


class _DummySparseConv(nn.Module):
    def __init__(self, outdim: int) -> None:
        super().__init__()
        self.outdim = outdim

    def forward(self, sparse_tensor: _DummySparseTensor) -> torch.Tensor:  # type: ignore[override]
        feats = sparse_tensor.feats
        if feats.ndim == 1:
            feats = feats.unsqueeze(-1)
        if feats.shape[1] != self.outdim:
            out = torch.zeros(feats.shape[0], self.outdim, device=feats.device)
            cols = min(feats.shape[1], self.outdim)
            out[:, :cols] = feats[:, :cols]
            return out
        return feats


def _dummy_construct_sparse_tensor(raw_coords, feats, Bbx_max, Bbx_min, voxel_size, device):
    vol_dim = torch.tensor([2, 2, 2], device=device)
    return _DummySparseTensor(raw_coords, feats), vol_dim, raw_coords


def _dummy_sparse_to_dense_volume(sparse_tensor: torch.Tensor, coords: torch.Tensor, vol_dim: torch.Tensor) -> torch.Tensor:
    device = coords.device
    feat_dim = sparse_tensor.shape[-1]
    shape = (int(vol_dim[0].item()), int(vol_dim[1].item()), int(vol_dim[2].item()), feat_dim)
    return torch.ones(shape, device=device) * 2.0


class _DummyImageExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros(x.shape[0], 1, 1, 2, device=x.device)


class _DummyAlphaT:
    def __init__(self, feat_dim: int = 2) -> None:
        self.feat_dim = feat_dim

    def render_rgb_only(self, gaussians, cameras, height: int, width: int):
        return [torch.zeros(height, width, 3, device=gaussians["means"].device) for _ in cameras]

    def render_and_backproject_streaming(self, gaussians, cameras, features_2d, height, width, num_gaussians, backprojector):
        vals = torch.arange(num_gaussians, device=gaussians["means"].device, dtype=torch.float32).unsqueeze(1)
        return vals.repeat(1, self.feat_dim)


def _make_config(use_2d: bool = False):
    return OmegaConf.create(
        {
            "model": {
                "use_2d_features": use_2d,
                "feat_2d_channels": 2,
                "feat_2d_downscale": 1,
                "sparseConv_outdim": 4,
                "voxel_size": 1.0,
                "bbx_min": [0.0, 0.0, 0.0],
                "bbx_max": [2.0, 2.0, 2.0],
            },
            "optimizer": {"lr": 1e-3},
            "training": {},
        }
    )


def _make_trainer(use_2d: bool = False) -> StreetForwardTrainer:
    cfg = _make_config(use_2d)
    return StreetForwardTrainer(
        config=cfg,
        device=torch.device("cpu"),
        renderer=lambda **kwargs: None,
        sparse_conv=_DummySparseConv(outdim=int(cfg.model.sparseConv_outdim)),
        construct_sparse_tensor_fn=_dummy_construct_sparse_tensor,
        sparse_to_dense_volume_fn=_dummy_sparse_to_dense_volume,
    )


def _make_node_states(device: torch.device):
    num_sh_rest = 3  # sh_degree = 1 -> (1+1)^2 = 4 bases, rest = 3

    means_bg = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    bg = NodeStateBackground(
        means=means_bg,
        scales_log=torch.zeros_like(means_bg),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device),
        opacity_logit=torch.zeros((2, 1), device=device),
        sh_dc=torch.zeros((2, 3), device=device),
        sh_rest=torch.zeros((2, num_sh_rest, 3), device=device),
    )

    means_rigid = torch.tensor(
        [
            [0.5, 0.5, 0.5],  # visible + in crop
            [3.0, 3.0, 3.0],  # visible but out of crop
            [0.5, 0.5, 0.5],  # in crop but invisible
        ],
        device=device,
    )
    rigid = NodeStateRigid(
        means=means_rigid,
        scales_log=torch.zeros_like(means_rigid),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1).to(device),
        opacity_logit=torch.zeros((3, 1), device=device),
        sh_dc=torch.zeros((3, 3), device=device),
        sh_rest=torch.zeros((3, num_sh_rest, 3), device=device),
        point_ids=torch.tensor([[0], [1], [2]], device=device),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3], device=device),
        instances_trans=torch.zeros((1, 3, 3), device=device),
        instances_fv=torch.tensor([[True, True, False]], device=device),
        instance_ids=[0, 1, 2],
        frame_ids=[0],
        cur_frame=0,
    )

    means_distant = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]], device=device)
    distant = NodeStateDistant(
        means=means_distant,
        scales_log=torch.zeros_like(means_distant),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1).to(device),
        opacity_logit=torch.zeros((2, 1), device=device),
        sh_dc=torch.zeros((2, 3), device=device),
        sh_rest=torch.zeros((2, num_sh_rest, 3), device=device),
    )

    return bg, rigid, distant


def test_prepare_all_gaussians_orders_bg_rigid_distant():
    trainer = _make_trainer()
    bg, rigid, distant = _make_node_states(trainer.device)

    gaussians, n_bg, n_rigid, n_distant = trainer._prepare_all_gaussians(
        node_state_bg=bg,
        node_state_rigid=rigid,
        node_state_distant=distant,
        source_frame_idx=0,
    )

    assert (n_bg, n_rigid, n_distant) == (2, 3, 2)
    expected_means = torch.cat([bg.means, rigid.means, distant.means], dim=0)
    assert torch.allclose(gaussians["means"], expected_means)


def test_compute_2d_features_all_respects_alignment_and_visibility():
    trainer = _make_trainer()
    trainer.use_2d_features = True
    trainer.image_feature_extractor = _DummyImageExtractor()
    trainer.alpha_t_extractor = _DummyAlphaT(feat_dim=2)
    trainer.feature_backprojector = object()

    bg, rigid, distant = _make_node_states(trainer.device)
    rigid_visible_mask = torch.tensor([1, 0, 1], dtype=torch.bool, device=trainer.device)

    feat_bg, feat_rigid, feat_distant = trainer._compute_2d_features_all(
        node_state_bg=bg,
        node_state_rigid=rigid,
        node_state_distant=distant,
        source_views=[object()],
        source_images=[torch.zeros(2, 2, 3)],
        source_frame_idx=0,
        rigid_visible_mask=rigid_visible_mask,
    )

    assert feat_bg.shape == (2, 2)
    assert feat_rigid.shape == (3, 2)
    assert feat_distant.shape == (2, 2)

    assert torch.allclose(feat_bg, torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=trainer.device))
    assert torch.allclose(feat_rigid, torch.tensor([[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]], device=trainer.device))
    assert torch.allclose(feat_distant, torch.tensor([[5.0, 5.0], [6.0, 6.0]], device=trainer.device))


def test_build_3d_feature_volume_masks_invisible_and_out_of_crop():
    trainer = _make_trainer()
    bg, rigid, _ = _make_node_states(trainer.device)

    feat_bg, feat_rigid, visible_mask, in_crop_mask = trainer._build_3d_feature_volume(
        node_state_bg=bg,
        node_state_rigid=rigid,
        source_frame_idx=0,
    )

    assert torch.equal(visible_mask, torch.tensor([True, True, False], device=trainer.device))
    assert torch.equal(in_crop_mask, torch.tensor([True, False, True], device=trainer.device))

    assert feat_bg.shape == (2, 4)
    assert torch.allclose(feat_bg, torch.full((2, 4), 2.0, device=trainer.device))

    assert torch.allclose(feat_rigid[0], torch.full((4,), 2.0, device=trainer.device))
    assert torch.allclose(feat_rigid[1], torch.zeros(4, device=trainer.device))
    assert torch.allclose(feat_rigid[2], torch.zeros(4, device=trainer.device))


# --- Baseline feature recording and comparison tests ------------------------


def test_record_step_populates_feature_summaries():
    """record_step fills feat_*_summary when trainer has _last_feat_* set."""
    from utils.streetforward_baseline import record_step

    trainer = _make_trainer(use_2d=False)
    bg, rigid, distant = _make_node_states(trainer.device)

    trainer._last_feat_3d_bg = torch.zeros(2, 4, device=trainer.device)
    trainer._last_feat_3d_rigid = torch.zeros(3, 4, device=trainer.device)
    trainer._last_feat_3d_distant = None
    trainer._last_feat_2d_bg = None
    trainer._last_feat_2d_rigid = None
    trainer._last_feat_2d_distant = None
    trainer._last_feat_bg_input = torch.ones(2, 4, device=trainer.device) * 0.5
    trainer._last_feat_rigid_input = torch.ones(3, 4, device=trainer.device) * 0.3
    trainer._last_feat_distant_input = None
    trainer._last_offsets_bg = None
    trainer._last_offsets_rigid = None
    trainer._last_offsets_distant = None

    batch = {
        "scene_id": torch.tensor([0], device=trainer.device),
        "segment_id": torch.tensor([0], device=trainer.device),
        "targets": [{}],
    }
    result = {
        "node_state": bg,
        "node_state_rigid": rigid,
        "node_state_distant": distant,
        "total_loss": torch.tensor(0.5, device=trainer.device),
    }

    step = record_step(trainer, batch, result, 0)

    assert step.feat_3d_bg_summary is not None
    assert step.feat_3d_bg_summary["shape"] == [2, 4]
    assert step.feat_3d_rigid_summary is not None
    assert step.feat_3d_rigid_summary["shape"] == [3, 4]
    assert step.feat_3d_distant_summary is None
    assert step.feat_2d_bg_summary is None
    assert step.feat_bg_input_summary is not None
    assert step.feat_bg_input_summary["shape"] == [2, 4]
    assert step.feat_rigid_input_summary is not None
    assert step.feat_distant_input_summary is None


def test_compare_step_feature_summaries_backward_compat():
    """When baseline has no feat_*_summary keys, compare_step skips them (no failure)."""
    from utils.streetforward_baseline import compare_step

    base = {
        "step": 0,
        "scene_id": 0,
        "segment_id": 0,
        "total_loss": 1.0,
        "node_state_bg_summary": {"num_points": 2},
        "node_state_rigid_summary": {"num_points": 3},
        "node_state_distant_summary": {"num_points": 0},
        "offset_bg_summary": None,
        "offset_rigid_summary": None,
        "offset_distant_summary": None,
        "grad_norms": {"sparse_conv": 0.0, "mlp_offset_pos": 0.0, "mlp_conv": 0.0, "mlp_opacity": 0.0, "gaussion_decoder": 0.0},
    }
    cur = dict(base)
    cur["total_loss"] = 1.0

    ok, msg = compare_step(base, cur)
    assert ok, msg


def test_compare_step_feature_summaries_value_alignment():
    """When both steps have feat_3d_bg_summary, identical values pass; different values fail."""
    from utils.streetforward_baseline import compare_step

    summary_same = {"shape": [2, 4], "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5, "norm": 1.5}
    summary_diff = {"shape": [2, 4], "min": 0.0, "max": 2.0, "mean": 0.6, "std": 0.4, "norm": 1.8}

    def make_step(feat_3d_bg_summary):
        return {
            "step": 0,
            "scene_id": 0,
            "segment_id": 0,
            "total_loss": 1.0,
            "node_state_bg_summary": {"num_points": 2},
            "node_state_rigid_summary": {"num_points": 3},
            "node_state_distant_summary": {"num_points": 0},
            "offset_bg_summary": None,
            "offset_rigid_summary": None,
            "offset_distant_summary": None,
            "feat_3d_bg_summary": feat_3d_bg_summary,
            "grad_norms": {"sparse_conv": 0.0, "mlp_offset_pos": 0.0, "mlp_conv": 0.0, "mlp_opacity": 0.0, "gaussion_decoder": 0.0},
        }

    ok, _ = compare_step(make_step(summary_same), make_step(summary_same))
    assert ok

    ok, msg = compare_step(make_step(summary_same), make_step(summary_diff))
    assert not ok
    assert "feat_3d_bg" in msg or "mismatch" in msg
