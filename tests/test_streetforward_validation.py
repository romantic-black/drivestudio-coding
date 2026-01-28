import json
import sys
from pathlib import Path

import torch

# Ensure project root on path for repro_step import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from repro_step import bind_feature_capture, compare_payload, run_once
from tests.streetforward_validation_utils import (
    build_minimal_batch,
    build_stub_trainer,
    collect_grad_norms,
    make_test_rigid_state,
    _normalize_quat,
    _quat_multiply,
    set_seed,
)


def test_geometry_transforms():
    device = torch.device("cpu")
    trainer = build_stub_trainer(device=device)
    rigid_state = make_test_rigid_state(device)

    # world transform: 90deg Z + trans (1,2,3) acting on (1,0,0) -> (1,3,3)
    world_pts = trainer._transform_rigid_to_world(rigid_state, rigid_state.means)
    expected = torch.tensor([[1.0, 3.0, 3.0]], device=device)
    assert torch.allclose(world_pts, expected, atol=1e-5)

    # offsets world -> local should rotate only (vector, no translation)
    offsets_world = {
        "offset_pos": torch.tensor([[1.0, 0.0, 0.0]], device=device),
        "offset_scales": torch.zeros(1, 3, device=device),
        "offset_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "offset_opacity": torch.zeros(1, 1, device=device),
        "offset_sh": torch.zeros(1, 3, device=device),
    }
    offsets_local = trainer._transform_offsets_world_to_local(rigid_state, offsets_world, frame_idx=0)
    expected_local = torch.tensor([[0.0, -1.0, 0.0]], device=device)
    assert torch.allclose(offsets_local["offset_pos"], expected_local, atol=1e-5)

    # quaternion composition: q_inst * q_local
    q_local = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=device)  # 90deg around X
    q_world = trainer._transform_rigid_quats_to_world(rigid_state, q_local)
    q_expected = _normalize_quat(_quat_multiply(rigid_state.instances_quats[0], q_local))
    assert torch.allclose(q_world, q_expected, atol=1e-5)


def test_contracts_and_no_nan():
    device = torch.device("cpu")
    set_seed(42)
    trainer = build_stub_trainer(device=device)
    bind_feature_capture(trainer)
    batch = build_minimal_batch(device)
    result = trainer.train_iter(batch, apply_update=False, update_state=False)

    loss = result["total_loss"]
    assert torch.isfinite(loss)
    offsets = trainer._last_offsets_bg
    assert offsets is not None
    assert torch.isfinite(offsets["offset_pos"]).all()
    assert offsets["offset_pos"].abs().max() <= trainer.offset_max + 1e-6

    # recorded feature stats have no NaN/Inf
    feat_bg = getattr(trainer, "_debug_feat_bg")
    assert torch.isfinite(feat_bg).all()


def test_gradient_connectivity_counts():
    device = torch.device("cpu")
    set_seed(42)
    trainer = build_stub_trainer(device=device)
    bind_feature_capture(trainer)
    batch = build_minimal_batch(device)
    trainer.train_iter(batch, apply_update=False, update_state=False)

    patterns = [
        "sparse_conv.",
        "mlp_offset_pos.",
        "mlp_conv.",
        "mlp_opacity.",
        "gaussion_decoder.",
    ]
    _, counts = collect_grad_norms(trainer, patterns, eps=1e-12)
    assert counts["matched_param_count"] > 0
    assert counts["none_grad_count"] == 0


def test_snapshot_matches_golden():
    device = torch.device("cpu")
    set_seed(42)
    current = run_once(device)
    golden_path = Path("docs/trainers/golden/streetforward_step1_stub_cpu.json")
    if not golden_path.exists():
        pytest.fail(
            "Golden missing. Generate with: "
            "python repro_step.py --backend stub --device cpu --seed 42 "
            "--save-golden docs/trainers/golden/streetforward_step1_stub_cpu.json"
        )
    golden = json.loads(golden_path.read_text())
    compare_payload(current, golden, rel_tol=1e-2, abs_tol=1e-4)
