import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse lightweight stubs from the feature alignment tests
from tests.test_streetforward_feature_alignment import _make_trainer, _make_node_states
from models.streetforward.node_state_mixin import RigidMasks


def _zero_offsets(num_points: int, device: torch.device):
    return {
        "offset_pos": torch.zeros(num_points, 3, device=device),
        "offset_scales": torch.zeros(num_points, 3, device=device),
        "offset_opacity": torch.zeros(num_points, 1, device=device),
    }


def _dummy_render_params(num_points: int, device: torch.device):
    return {
        "means_r": torch.zeros(num_points, 3, device=device, requires_grad=True),
        "scales_r": torch.zeros(num_points, 3, device=device, requires_grad=True),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points, device=device, requires_grad=True),
        "opacities_r": torch.ones(num_points, device=device, requires_grad=True),
        "colors_r": torch.zeros(num_points, 4, 3, device=device, requires_grad=True),
    }


def test_strict_proxy_grad_raises_on_none():
    trainer = _make_trainer()
    trainer._strict_proxy_grad_active = True
    trainer._strict_checks_active = True

    render_params_bg = _dummy_render_params(num_points=1, device=trainer.device)
    proxies_bg = {
        "means_p": render_params_bg["means_r"].detach().clone().requires_grad_(True),
        "scales_p": render_params_bg["scales_r"].detach().clone().requires_grad_(True),
        "quats_p": render_params_bg["quats_r"].detach().clone().requires_grad_(True),
        "opacities_p": render_params_bg["opacities_r"].detach().clone().requires_grad_(True),
        "colors_p": render_params_bg["colors_r"].detach().clone().requires_grad_(True),
    }

    with pytest.raises(RuntimeError):
        trainer._backward_to_render_params(
            render_params_bg=render_params_bg,
            render_params_rigid=None,
            render_params_distant=None,
            proxies_bg=proxies_bg,
            proxies_rigid=None,
            proxies_distant=None,
        )


def test_sentinel_metrics_cover_masks_and_volume():
    trainer = _make_trainer()
    trainer.sentinel_enabled = True
    bg, rigid, distant = _make_node_states(trainer.device)

    masks = RigidMasks(
        mask_src_rigid=torch.tensor([True, False, True], device=trainer.device),
        mask_tgt_rigid=[torch.tensor([True, False, False], device=trainer.device)],
        mask_any_tgt_rigid=torch.tensor([True, False, True], device=trainer.device),
        mask_update_rigid=torch.tensor([True, False, True], device=trainer.device),
        idx_tgt_rigid=[torch.tensor([0, 2], device=trainer.device)],
        idx_src_rigid=torch.tensor([0, 2], device=trainer.device),
    )

    trainer._last_vol_dim_prod = 27
    trainer._last_dense_elements_est = 270

    render_params_bg = {
        "means_r": bg.means,
        "scales_log_r": bg.scales_log,
        "quats_r": bg.quats,
        "opacities_r": torch.sigmoid(bg.opacity_logit).squeeze(-1),
    }

    trainer._collect_sentinel_metrics(
        targets=[{"frame_idx": 0, "view": None, "gt_image": torch.zeros(1, 1, 3)}],
        node_state_bg=bg,
        node_state_rigid=rigid,
        node_state_distant=distant,
        masks=masks,
        render_params_bg=render_params_bg,
        render_params_rigid=None,
        render_params_distant=None,
        offsets_bg=_zero_offsets(bg.means.shape[0], trainer.device),
        offsets_rigid_world=None,
        offsets_distant=None,
    )

    metrics = trainer._last_sentinel_metrics
    assert metrics["mask_update_rigid_mean"] == pytest.approx(2 / 3)
    assert metrics["idx_tgt_rigid_mean"] == pytest.approx(2.0)
    assert metrics["vol_dim_prod"] == 27.0
    assert metrics["dense_elements_est"] == 270.0
    assert "bg_opacities_min" in metrics


def test_record_volume_stats_respects_limit():
    trainer = _make_trainer()
    trainer.sentinel_max_dense_elements = 1
    with pytest.raises(RuntimeError):
        trainer._record_volume_stats(vol_dim=torch.tensor([2, 2, 2]), feat_dim=2)


def test_h_cache_resets_when_signature_changes():
    trainer = _make_trainer()
    bg, rigid, _ = _make_node_states(trainer.device)
    key = (0, 0)
    # Seed cache with wrong size and signature
    trainer.h_cache_bg[key] = torch.ones(3, trainer.offset_gru_hidden_dim, device=trainer.device)
    trainer._h_cache_signatures["bg"] = {key: (3,)}

    h = trainer._get_or_init_hidden(
        trainer.h_cache_bg, key, num_points=bg.means.shape[0], node_state=bg, node_type="bg"
    )
    assert h.shape[0] == bg.means.shape[0]
    assert torch.allclose(h, torch.zeros_like(h))
    assert trainer._h_cache_signatures["bg"][key] == trainer._cache_signature(bg)


def test_sentinel_alert_on_nan_raises():
    trainer = _make_trainer()
    trainer.sentinel_enabled = True
    trainer.sentinel_alert_on_nan = True
    trainer._last_sentinel_metrics = {"bg_opacities_min": float("nan")}
    with pytest.raises(RuntimeError):
        trainer._maybe_alert_on_sentinel()
