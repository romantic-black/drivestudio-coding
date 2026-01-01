import importlib.util
import os
import types

import numpy as np
import torch
from omegaconf import OmegaConf

_MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trainers", "streetforward.py")
_SPEC = importlib.util.spec_from_file_location("streetforward", _MODULE_PATH)
streetforward = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(streetforward)

StreetForwardTrainer = streetforward.StreetForwardTrainer
get_viewmat = streetforward.get_viewmat
_sh_to_rgb = streetforward._sh_to_rgb


class DummyView(types.SimpleNamespace):
    """Simple container to mimic camera objects used by the trainer."""

    pass


def _build_config():
    return OmegaConf.create(
        {
            "model": {
                "sparseConv_outdim": 4,
                "offset_max": 0.05,
                "sh_degree": 1,
                "voxel_size": 0.5,
                "max_iterations": 1,
                "bbx_min": [-1.0, -1.0, -1.0],
                "bbx_max": [1.0, 1.0, 1.0],
            },
            "optimizer": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.0},
        }
    )


def _make_batch(device: torch.device, image_value: float = 0.0):
    scene_id, segment_id = 0, 0
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [-0.2, 0.1, -0.1]], dtype=np.float32)
    colors = np.array([[0.3, 0.4, 0.5], [0.6, 0.4, 0.2], [0.1, 0.9, 0.3]], dtype=np.float32)
    pointcloud = {"background": np.concatenate([points, colors], axis=1)}

    c2w = torch.eye(4, device=device).unsqueeze(0)
    k = torch.eye(3, device=device).unsqueeze(0)
    view = DummyView(camtoworlds=c2w, Ks=k)
    gt_image = torch.full((2, 2, 3), image_value, device=device)

    return {
        "scene_id": scene_id,
        "segment_id": segment_id,
        "pointcloud": pointcloud,
        "target_views": [view, view],
        "gt_images": [gt_image.clone(), gt_image.clone()],
    }


def test_node_state_detached_after_update():
    device = torch.device("cpu")
    config = _build_config()
    trainer = StreetForwardTrainer(config, device=device)
    batch = _make_batch(device)

    trainer.train_iter(batch, apply_update=False, update_state=True)
    state = trainer.node_states[(0, 0)]

    assert not state.means.requires_grad
    assert not state.scales_log.requires_grad
    assert not state.quats.requires_grad
    assert not state.opacity_logit.requires_grad
    assert not state.sh_dc.requires_grad
    assert not state.sh_rest.requires_grad
    assert state.means.shape[1] == 3


def _run_direct_loss(trainer: StreetForwardTrainer, batch: dict) -> torch.Tensor:
    key, node_state = trainer._get_or_init_node_state(batch)
    view_count = max(len(batch["target_views"]), 1)

    means_s = node_state.means
    anchor_rgb = _sh_to_rgb(node_state.sh_dc)
    sparse_feat, vol_dim, valid_coords = trainer.construct_sparse_tensor(
        raw_coords=means_s.clone(),
        feats=anchor_rgb,
        Bbx_max=trainer.bbx_max,
        Bbx_min=trainer.bbx_min,
        voxel_size=trainer.voxel_size,
    )
    feat_3d = trainer.sparse_conv(sparse_feat)
    dense_volume = trainer.sparse_to_dense_volume(
        sparse_tensor=feat_3d,
        coords=valid_coords,
        vol_dim=vol_dim,
    ).unsqueeze(0)
    dense_volume = dense_volume.permute(0, 4, 3, 1, 2)

    grid_coords = trainer.get_grid_coords(means_s, trainer.bbx_min, vol_dim, trainer.voxel_size)
    feat_3d_crop = trainer.interpolate_features(grid_coords, dense_volume)
    offsets = trainer._predict_offsets(feat_3d_crop)
    render_params = trainer._render_params_from_offsets(node_state, offsets)

    loss = torch.tensor(0.0, device=trainer.device)
    for view, gt_img in zip(batch["target_views"], batch["gt_images"]):
        viewmat = get_viewmat(view.camtoworlds)
        render, _, _ = trainer.renderer(
            means=render_params["means_r"],
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=viewmat,
            Ks=view.Ks,
            width=gt_img.shape[1],
            height=gt_img.shape[0],
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=trainer.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )
        rgb = render[:, ..., :3].squeeze(0)
        loss = loss + trainer.compute_loss(rgb, gt_img) / view_count
    trainer.node_states[key] = node_state.detach_clone()
    return loss


def test_proxy_and_direct_gradients_match():
    torch.manual_seed(42)
    device = torch.device("cpu")
    config = _build_config()
    trainer = StreetForwardTrainer(config, device=device)
    batch = _make_batch(device, image_value=0.1)

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.train_iter(batch, apply_update=False, update_state=False)
    proxy_grad = trainer.mlp_offset_pos[0].weight.grad.clone()

    trainer.optimizer.zero_grad(set_to_none=True)
    loss_direct = _run_direct_loss(trainer, batch)
    loss_direct.backward()
    direct_grad = trainer.mlp_offset_pos[0].weight.grad.clone()

    assert proxy_grad is not None and direct_grad is not None
    assert torch.allclose(proxy_grad, direct_grad, atol=1e-6, rtol=1e-4)
