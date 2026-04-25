from __future__ import annotations

import torch
import pytest

from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3


class _DummyStage4_3:
    def __init__(self) -> None:
        self.training = True
        self.infer_batch = None
        self.forward_batch = None
        self.writeback_out = None
        self.export_args = None
        self.restored = None

    def ensure_runtime_state_from_batch(self, batch):
        self.infer_batch = batch

    def _batch_key(self, batch):
        _ = batch
        return (7, 9)

    def _snapshot_runtime_state(self, key):
        _ = key
        return {"snap": 1}

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward(self, batch):
        self.forward_batch = batch
        return {"_cache_key": (7, 9)}

    def _writeback_node_states_from_out(self, out):
        self.writeback_out = out

    def export_3dgs_state(self, batch, *, include_hidden: bool, rigid_export_frame_idx: int):
        self.export_args = {
            "batch": batch,
            "include_hidden": bool(include_hidden),
            "rigid_export_frame_idx": int(rigid_export_frame_idx),
        }
        return {"ok": True}

    def _restore_runtime_state(self, key, snap):
        self.restored = {"key": key, "snap": snap}


def test_build_scene_representation_from_source_uses_all_source_views_as_targets():
    dummy = _DummyStage4_3()
    source_views = [
        {"cam_idx": 0},
        {"cam_idx": 1},
    ]
    source_images = [
        torch.zeros((4, 5, 3), dtype=torch.float32),
        torch.ones((4, 5, 3), dtype=torch.float32),
    ]
    source_sky = [torch.zeros((4, 5), dtype=torch.float32), torch.ones((4, 5), dtype=torch.float32)]
    source_ego = [torch.zeros((4, 5), dtype=torch.float32), torch.zeros((4, 5), dtype=torch.float32)]
    source_vd = [torch.zeros((4, 5, 3), dtype=torch.float32), torch.ones((4, 5, 3), dtype=torch.float32)]
    batch = {
        "source_views": source_views,
        "source_images": source_images,
        "source_sky_mask": source_sky,
        "source_egocar_mask": source_ego,
        "source_viewdirs": source_vd,
        "source_frame_idx": 123,
    }

    out = MinimalStreetForwardStage4_3.build_scene_representation_from_source(
        dummy,
        batch,
        allow_hidden_cache_update=False,
        allow_node_state_writeback=False,
    )

    infer_batch = dummy.infer_batch
    assert infer_batch is not None
    targets = infer_batch["targets"]
    assert len(targets) == 2
    assert [int(t["frame_idx"]) for t in targets] == [123, 123]
    assert [int(t["cam_idx"]) for t in targets] == [0, 1]
    assert targets[0]["view"] is source_views[0]
    assert targets[1]["view"] is source_views[1]
    assert targets[0]["gt_image"] is source_images[0]
    assert targets[1]["gt_image"] is source_images[1]
    assert targets[0]["sky_mask"] is source_sky[0]
    assert targets[1]["sky_mask"] is source_sky[1]
    assert targets[0]["egocar_mask"] is source_ego[0]
    assert targets[1]["egocar_mask"] is source_ego[1]
    assert targets[0]["viewdirs"] is source_vd[0]
    assert targets[1]["viewdirs"] is source_vd[1]

    assert dummy.export_args is not None
    assert int(dummy.export_args["rigid_export_frame_idx"]) == 123
    assert dummy.restored == {"key": (7, 9), "snap": {"snap": 1}}
    assert out["cache_key"] == (7, 9)
    assert out["base_batch"]["targets"] == targets


def test_build_scene_representation_from_source_fast_fails_on_source_len_mismatch():
    dummy = _DummyStage4_3()
    batch = {
        "source_views": [{"cam_idx": 0}, {"cam_idx": 1}],
        "source_images": [torch.zeros((4, 5, 3), dtype=torch.float32)],
        "source_frame_idx": 1,
    }
    with pytest.raises(ValueError, match="len\\(source_views\\) == len\\(source_images\\)"):
        MinimalStreetForwardStage4_3.build_scene_representation_from_source(
            dummy,
            batch,
            allow_hidden_cache_update=False,
            allow_node_state_writeback=False,
        )

