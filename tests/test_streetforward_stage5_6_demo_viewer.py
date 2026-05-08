from __future__ import annotations

import types

import numpy as np
import torch
from omegaconf import OmegaConf

from streetforward_eval.runner import StreetForwardBatchEvalRunner
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_demo_scheduler import build_stage5_demo_scheduler_from_cfg


class _SegmentIndex:
    def __init__(self, *, num_frames: int = 12, num_cams: int = 2):
        self.frame_indices = list(range(num_frames))
        self.train_frame_set = set(range(num_frames))
        self.num_cams = int(num_cams)


class _TinyDataset:
    def __init__(self):
        self.sidx = _SegmentIndex()

    def list_training_scene_ids(self):
        return [10]

    def list_segment_ids(self, scene_id: int):
        assert int(scene_id) == 10
        return [0, 1]

    def get_segment_index(self, scene_id: int, segment_id: int):
        assert int(scene_id) == 10
        assert int(segment_id) in (0, 1)
        return self.sidx

    def _role_dict(self, refs):
        n = len(refs)
        image = torch.zeros((n, 4, 5, 3), dtype=torch.float32)
        extr = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        intr = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        return {
            "image": image,
            "extrinsics": extr,
            "intrinsics": intr,
            "frame_indices": torch.tensor([int(f) for f, _c in refs], dtype=torch.long),
            "cam_indices": torch.tensor([int(c) for _f, c in refs], dtype=torch.long),
            "sky_mask": torch.zeros((n, 4, 5, 1), dtype=torch.float32),
            "egocar_mask": torch.zeros((n, 4, 5, 1), dtype=torch.float32),
        }

    def _assemble_segment_batch_from_image_refs(
        self,
        scene_id,
        segment_id,
        source_image_refs,
        target_image_refs,
        aux_image_refs=None,
        include_test=False,
        test_image_refs=None,
        enforce_target0_equals_source=True,
        target_ref_purpose="train",
    ):
        _ = (aux_image_refs, include_test, test_image_refs, enforce_target0_equals_source, target_ref_purpose)
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "pointcloud": np.zeros((0, 6), dtype=np.float32),
            "source": self._role_dict(source_image_refs),
            "target": self._role_dict(target_image_refs),
            "request_meta": {},
        }


def _cfg(*, steps_per_input: int = 2, switch: int = 1):
    return OmegaConf.create(
        {
            "model": {"stage": "5_6", "production_training": True},
            "training": {"seed": 0},
            "batch_eval": {
                "cameras": {"ids": [0, 1], "names": ["cam0", "cam1"]},
                "update_cameras": {"ids": [0, 1], "names": ["cam0", "cam1"]},
                "runtime": {"update_node_state": True, "update_hidden_state": True},
                "stage5_6_eval": {
                    "nearby_policy": "adjacent_non_input",
                    "nearby_role_name": "near_random",
                    "allow_partial_nearby": True,
                },
                "history": {"record_support_residual_on_input_exit": True, "record_each_step": False},
            },
            "demo": {
                "mode": "segment_finetune_train",
                "scheduler": {
                    "type": "eval_v8_stage5_6",
                    "scene_ids": [10],
                    "initial_scene_id": 10,
                    "initial_segment_id": 0,
                    "initial_sequence_start_pos": 0,
                    "sequence_length": 10,
                    "input_offsets": [1, 3, 5, 7, 9],
                    "eval_offsets": "all",
                    "steps_per_input": int(steps_per_input),
                    "block_order": "step_major",
                    "step_major_switch_interval_steps": int(switch),
                    "max_target_frames_including_source": 3,
                    "window_policy": "sliding",
                    "stride": 1,
                    "require_full_window": True,
                },
            },
        }
    )


def test_stage5_6_demo_eval_cursor_matches_batcheval_order():
    cfg = _cfg(steps_per_input=3, switch=2)
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    expected = StreetForwardBatchEvalRunner._iter_block_visit_order(
        num_blocks=5,
        steps_per_block=3,
        block_order="step_major",
        step_major_switch_interval_steps=2,
    )
    seen = []
    for _ in expected:
        info = sched.materialize_current_batch_without_advance()["_scheduler_v8_aligned_info"]
        seen.append(int(info["block_idx_in_episode"]))
        sched.mark_current_block_updated()
    assert seen == expected
    assert sched.is_episode_done()


def test_stage5_6_demo_batch_metadata_has_near_random_roles():
    sched = build_stage5_demo_scheduler_from_cfg(_cfg(), _TinyDataset(), device=torch.device("cpu"))
    batch = sched.materialize_current_batch_without_advance()
    rm = batch["request_meta"]
    assert rm["near_random_frame_indices"] == [0, 2]
    assert rm["target_frame_roles"] == ["source", "near_random", "near_random"]
    assert len(rm["target_image_roles"]) == len(rm["target_image_refs"])
    assert rm["target_image_roles"] == ["source", "source", "near_random", "near_random", "near_random", "near_random"]


def test_stage5_6_demo_can_select_segment_and_window():
    sched = build_stage5_demo_scheduler_from_cfg(_cfg(), _TinyDataset(), device=torch.device("cpu"))
    sched.set_scope(10, 1)
    assert sched.get_current_info()["segment_id"] == 1
    sched.set_sequence_start_pos(1)
    info = sched.materialize_current_batch_without_advance()["_scheduler_v8_aligned_info"]
    assert int(info["scene_id"]) == 10
    assert int(info["segment_id"]) == 1
    assert info["block_current_source_frame_indices"] == [2, 4, 6, 8, 10]


class _Trainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.seen = {}
        self.optimizer = types.SimpleNamespace(global_step=123, state={}, zero_grad=lambda *a, **k: None)

    def train_step(
        self,
        batch,
        step=None,
        profile_phase_timing=False,
        sync_cuda_timing=False,
        scheduler_node_sync=None,
        runtime_policy=None,
    ):
        _ = (batch, step, profile_phase_timing, sync_cuda_timing)
        self.seen["scheduler_node_sync"] = dict(scheduler_node_sync or {})
        self.seen["runtime_policy"] = runtime_policy
        return {"loss": 1.0}

    def record_block_history(self, batch, event=None):
        self.seen["history_event"] = dict(event or {})
        return {"history_recorded": 1.0}

    def reset_node_state(self):
        self.seen["reset_node_state"] = True


def test_stage5_6_demo_controller_uses_runtime_policy_and_history_on_block_exit():
    cfg = _cfg(steps_per_input=1, switch=1)
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    trainer = _Trainer()
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=trainer,
        device=torch.device("cpu"),
        stage="5_6",
    )
    controller.prime()
    stats = controller.step_current_block_once()
    assert trainer.seen["scheduler_node_sync"]["segment_local_step"] == 1
    assert trainer.seen["runtime_policy"].do_backward is True
    assert trainer.seen["runtime_policy"].do_optimizer_step is True
    assert trainer.seen["history_event"]["block_idx_in_episode"] == 0
    assert stats["history_recorded"] == 1.0


def test_stage5_6_demo_scope_change_resets_runtime_and_params():
    cfg = _cfg()
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    trainer = _Trainer()
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=trainer,
        device=torch.device("cpu"),
        stage="5_6",
    )
    controller.prime()
    with torch.no_grad():
        trainer.weight.fill_(5.0)
    stats = controller.set_sequence_start_pos(1)
    assert bool(trainer.seen["reset_node_state"]) is True
    assert float(trainer.weight.item()) == 1.0
    assert stats["reset_training_parameters"] == 1.0
