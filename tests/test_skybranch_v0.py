from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from models.streetforward.sky_branch import FrozenStreetForwardSceneProvider, MinimalSkyBranchTrainer, SceneRenderPack, SkyBranchV0
from models.streetforward.sky_branch.sky_render_utils import rotation_only_viewmat_from_view


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "sky": {
                "center_local": [0.0, 0.0, 0.0],
                "radius": 2.0,
                "resolution": 2,
                "hemisphere": True,
                "hemisphere_up": [0.0, -1.0, 0.0],
            },
            "sky_branch": {
                "sh_degree": 1,
                "feature_dim": 8,
                "hidden_dim": 8,
                "feature_extractor": {"in_channels": 8, "hidden_dim": 8, "output_dim": 8, "num_blocks": 1},
                "lifting": {"support_min": 1.0e-4, "weight_threshold": 0.0, "use_sky_core_mask": True, "sky_core_erode_kernel": 1},
                "init": {"opacity_init": 0.7, "scale_init": {"isotropic_log_value": -1.5}},
                "eta": {"scales": 0.03, "opacity": 0.2, "sh_dc": 0.05, "sh_rest": 0.02},
            },
            "loss": {
                "eps": 1.0e-3,
                "comp_weight": 1.0,
                "sky_direct_weight": 0.2,
                "sky_alpha_weight": 0.05,
                "sky_core_erode_kernel": 1,
                "semantic_weight": {"sky_core": 1.0, "sky_boundary": 0.2, "non_sky": 0.05},
            },
            "optimizer": {"type": "adamw", "lr": 1.0e-3, "eps": 1.0e-8, "weight_decay": 0.0},
            "training": {"amp": False, "grad_clip_norm": 1.0},
        }
    )


def _view() -> SimpleNamespace:
    return SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3).unsqueeze(0))


def _fake_renderer(**kwargs):
    height = int(kwargs["height"])
    width = int(kwargs["width"])
    viewmats = kwargs["viewmats"]
    colors = kwargs["colors"]
    opacities = kwargs["opacities"]
    rgb = torch.sigmoid(colors.mean(dim=(0, 1))).view(1, 1, 1, 3).expand(int(viewmats.shape[0]), height, width, 3)
    alpha = opacities.mean().sigmoid().view(1, 1, 1, 1).expand(int(viewmats.shape[0]), height, width, 1)
    return rgb, alpha, {}


class _FakeExtractor:
    def render_and_backproject_streaming_fused_multi_camera(self, **kwargs):
        mask = kwargs["source_pair_valid_mask"]
        assert mask.dtype == torch.bool
        assert tuple(mask.shape) == (1, 2, 2)
        feat2d = kwargs["features_2d"]
        n = int(kwargs["num_gaussians"])
        feat = feat2d.mean(dim=(0, 1, 2), keepdim=False).unsqueeze(0).expand(n, -1)
        support = torch.ones(n, device=feat2d.device, dtype=feat2d.dtype)
        return feat, support


class _FakeSceneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(()))
        self.version = 0
        self.render_calls = 0
        self.reset_calls = 0
        self.node_states_bg = {"state": torch.ones(1)}
        self.h_cache_bg = {"state": torch.ones(1)}

    def inference_step_from_train_batch(self, batch, scheduler_node_sync=None, runtime_policy=None):
        del batch, scheduler_node_sync, runtime_policy
        self.version += 1
        return {"loss": 0.0}

    def _render_scene_views_from_current_state(self, batch, items):
        del batch
        self.render_calls += 1
        rgbs = []
        alphas = []
        for item in items:
            gt = item["gt_image"]
            h, w = int(gt.shape[0]), int(gt.shape[1])
            rgbs.append(torch.zeros(h, w, 3) + float(self.version))
            alphas.append(torch.zeros(h, w, 1) + float(self.version) / 10.0)
        return torch.stack(rgbs, dim=0), torch.stack(alphas, dim=0)

    def reset_node_state(self):
        self.reset_calls += 1


def test_rotation_only_viewmat_zeros_translation():
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([3.0, 4.0, 5.0])
    view = SimpleNamespace(camtoworlds=c2w, Ks=torch.eye(3).unsqueeze(0))
    viewmat = rotation_only_viewmat_from_view(view)
    assert torch.allclose(viewmat[..., :3, 3], torch.zeros_like(viewmat[..., :3, 3]))


def test_scene_provider_render_pack_uses_render_alpha_maps_and_single_updated_state():
    model = _FakeSceneModel()
    provider = FrozenStreetForwardSceneProvider(device=torch.device("cpu"), model=model)
    img = torch.zeros(2, 2, 3)
    batch = {
        "source_views": [_view()],
        "source_images": [img],
        "source_frame_idx": 4,
        "targets": [{"view": _view(), "gt_image": img, "frame_idx": 5}],
    }
    pack = provider.render_batch(batch, scheduler_node_sync={"U": 1, "segment_local_step": 1, "reset_after_block": True})
    assert isinstance(pack, SceneRenderPack)
    assert tuple(pack.source_alpha.shape) == (1, 2, 2, 1)
    assert tuple(pack.target_alpha.shape) == (1, 2, 2, 1)
    assert torch.allclose(pack.source_alpha, pack.target_alpha)
    assert model.version == 1
    assert model.reset_calls == 0
    provider.apply_pending_reset()
    assert model.reset_calls == 1


def test_scene_provider_render_scene_views_does_not_modify_model_state():
    model = _FakeSceneModel()
    model.version = 7
    provider = FrozenStreetForwardSceneProvider(device=torch.device("cpu"), model=model)
    img = torch.zeros(2, 2, 3)
    batch = {"targets": [{"view": _view(), "gt_image": img, "frame_idx": 1}]}
    before = (model.version, {k: v.clone() for k, v in model.h_cache_bg.items()})
    rgb, alpha = provider.render_scene_views(batch, batch["targets"])
    after = (model.version, model.h_cache_bg)
    assert tuple(rgb.shape) == (1, 2, 2, 3)
    assert tuple(alpha.shape) == (1, 2, 2, 1)
    assert before[0] == after[0]
    assert torch.equal(before[1]["state"], after[1]["state"])


def test_skybranch_lifting_mask_is_bool_image_resolution():
    branch = SkyBranchV0(_cfg(), torch.device("cpu"), renderer=_fake_renderer, alpha_t_extractor=_FakeExtractor())
    sky = torch.ones(1, 2, 2)
    valid = torch.tensor([[[True, False], [True, True]]])
    mask = branch.build_lifting_mask(sky, valid)
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (1, 2, 2)
    assert int(mask.sum().item()) == 3


def test_skybranch_forward_keeps_feature_extractor_grad_path():
    branch = SkyBranchV0(_cfg(), torch.device("cpu"), renderer=_fake_renderer, alpha_t_extractor=_FakeExtractor())
    img = torch.zeros(2, 2, 3)
    batch = {
        "scene_id": 1,
        "segment_id": 2,
        "source_views": [_view()],
        "source_images": [img],
        "source_sky_masks": [torch.ones(2, 2)],
        "targets": [{"view": _view(), "gt_image": torch.ones(2, 2, 3) * 0.8, "sky_mask": torch.ones(2, 2), "frame_idx": 1}],
    }
    scene_pack = SceneRenderPack(
        source_rgb=torch.zeros(1, 2, 2, 3),
        source_alpha=torch.zeros(1, 2, 2, 1),
        target_rgb=torch.zeros(1, 2, 2, 3),
        target_alpha=torch.zeros(1, 2, 2, 1),
    )
    out = branch.forward_scene_batch(batch, scene_pack)
    out.loss.backward()
    grads = [p.grad for p in branch.feature_extractor.parameters() if p.requires_grad]
    assert any(g is not None and float(g.abs().sum().item()) > 0.0 for g in grads)
    assert torch.equal(out.node_state_sky.means, branch.node_states_sky[(1, 2)].means)


def test_model_checkpoint_excludes_runtime_sky_state(tmp_path):
    cfg = _cfg()
    branch = SkyBranchV0(cfg, torch.device("cpu"), renderer=_fake_renderer, alpha_t_extractor=_FakeExtractor())
    branch.get_or_init_node_state({"scene_id": 1, "segment_id": 2})
    provider = FrozenStreetForwardSceneProvider(device=torch.device("cpu"), model=_FakeSceneModel())
    trainer = MinimalSkyBranchTrainer(cfg, torch.device("cpu"), scene_provider=provider, sky_branch=branch)
    path = tmp_path / "skybranch_model.pth"
    trainer.save_checkpoint(str(path), kind="model")
    payload = torch.load(path, map_location="cpu")
    assert payload["kind"] == "model"
    assert "sky_branch_state_dict" in payload
    assert "node_states_sky" not in payload
    assert "h_cache_sky" not in payload


def test_resume_checkpoint_restores_runtime_sky_state(tmp_path):
    cfg = _cfg()
    branch = SkyBranchV0(cfg, torch.device("cpu"), renderer=_fake_renderer, alpha_t_extractor=_FakeExtractor())
    state = branch.get_or_init_node_state({"scene_id": 1, "segment_id": 2})
    h = branch.get_or_init_hidden((1, 2), state)
    h.add_(3.0)
    provider = FrozenStreetForwardSceneProvider(device=torch.device("cpu"), model=_FakeSceneModel())
    trainer = MinimalSkyBranchTrainer(cfg, torch.device("cpu"), scene_provider=provider, sky_branch=branch)
    path = tmp_path / "skybranch_resume.pth"
    trainer.save_checkpoint(str(path), kind="resume")

    restored_branch = SkyBranchV0(cfg, torch.device("cpu"), renderer=_fake_renderer, alpha_t_extractor=_FakeExtractor())
    restored_provider = FrozenStreetForwardSceneProvider(device=torch.device("cpu"), model=_FakeSceneModel())
    restored_trainer = MinimalSkyBranchTrainer(cfg, torch.device("cpu"), scene_provider=restored_provider, sky_branch=restored_branch)
    payload = restored_trainer.load_resume_checkpoint(str(path))

    assert payload["kind"] == "resume"
    assert (1, 2) in restored_branch.node_states_sky
    assert (1, 2) in restored_branch.h_cache_sky
    assert torch.allclose(restored_branch.h_cache_sky[(1, 2)], torch.full_like(restored_branch.h_cache_sky[(1, 2)], 3.0))
