from __future__ import annotations

import types

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.train_scheduler_v8 import TrainSchedulerV8
from streetforward_eval.runner import StreetForwardBatchEvalRunner
from tools.streetforward_stage5_demo_controller import Stage5DemoController
from tools.streetforward_stage5_demo_scheduler import build_stage5_demo_scheduler_from_cfg


class _SegmentIndex:
    def __init__(self, *, num_keyframes: int = 8, frames_per_keyframe: int = 3, num_cams: int = 2):
        num_frames = int(num_keyframes) * int(frames_per_keyframe)
        self.frame_indices = list(range(num_frames))
        self.train_frame_set = set(range(num_frames))
        self.num_cams = int(num_cams)
        self.keyframe_indices = list(range(int(num_keyframes)))
        self.keyframe_to_frames = {
            int(k): [int(k * frames_per_keyframe + i) for i in range(int(frames_per_keyframe))]
            for k in self.keyframe_indices
        }
        self.frame_to_keyframe = {
            int(f): int(k)
            for k, frames in self.keyframe_to_frames.items()
            for f in frames
        }
        self.test_frame_indices = []


class _TinyDataset:
    def __init__(self):
        self.sidx = _SegmentIndex()
        self._initialized = False

    def initialize(self):
        self._initialized = True

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

    def get_segment_batch_from_image_refs(self, request, *, enforce_target0_equals_source=True):
        return self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            request.source_image_refs or [request.source_image_ref],
            request.target_image_refs,
            include_test=bool(getattr(request, "include_test", False)),
            test_image_refs=getattr(request, "test_image_refs", None),
            enforce_target0_equals_source=bool(enforce_target0_equals_source),
        )

    def create_train_scheduler_v8(self, **kwargs):
        return TrainSchedulerV8(dataset=self, **kwargs)


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


def _train_v8_cfg(
    *,
    steps_per_block: int = 2,
    switch: int = 1,
    stage: str = "5_6",
    scheduler_type: str = "train_v8_stage5_6",
):
    return OmegaConf.create(
        {
            "model": {"stage": str(stage), "production_training": True},
            "training": {"seed": 0},
            "data": {"train_scene_ids": [10]},
            "scheduler_v8": {
                "enable": True,
                "block": {"steps_per_block": int(steps_per_block)},
                "episode": {
                    "blocks_per_episode": 3,
                    "total_target_frames": 2,
                    "include_source_frame": True,
                    "target_policy": "visited_episode_frames",
                    "block_source_frame_policy": "fixed_once_per_episode",
                    "frame_within_keyframe_policy": "middle_frame",
                    "min_keyframes_required_policy": "skip_if_less_than_window",
                },
                "traversal": {
                    "mode": "linear_scene_segment",
                    "switch_after_episode": True,
                    "fixed_scene_id": None,
                    "fixed_segment_id": None,
                    "segment_order": "ascending",
                    "scene_order": "ascending",
                },
                "execution": {
                    "block_order": "step_major",
                    "step_major_switch_interval_steps": int(switch),
                    "reset_policy": "episode_end",
                },
                "preload": {
                    "emit_hints": False,
                    "warm_next_block_exact": False,
                    "warm_next_episode_chain": False,
                },
                "aux_feature_splat_targets": {"enable": False},
                "near_random_supervision": {
                    "enable": True,
                    "frames_per_block": 2,
                    "same_keyframe_only": True,
                    "insufficient_policy": "skip",
                    "sample_once_per_block": True,
                    "exclude_source_frame": True,
                    "exclude_existing_target_frames": True,
                    "camera_policy": "all_cams",
                    "role_name": "near_random",
                },
            },
            "batch_eval": {
                "history": {"record_support_residual_on_input_exit": True, "record_each_step": False},
            },
            "demo": {
                "mode": "segment_finetune_train",
                "scheduler": {
                    "type": str(scheduler_type),
                    "scene_ids": [10],
                    "initial_scene_id": 10,
                    "initial_segment_id": 0,
                    "initial_sequence_start_pos": 0,
                    "wrap_episode": True,
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


def test_stage5_6_eval_demo_honors_batch_eval_start_at_when_demo_start_is_unspecified():
    cfg = _cfg()
    cfg.batch_eval.dataset = {"start_at": {"scene_id": 10, "segment_id": 1, "frame_id": 4}}
    cfg.demo.scheduler.initial_segment_id = None
    cfg.demo.scheduler.initial_sequence_start_pos = None
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    info = sched.get_current_info()
    assert int(info["segment_id"]) == 1
    assert int(info["sequence_start_pos"]) == 4
    assert sched.list_sequence_start_positions()[0] == 4


def test_stage5_6_train_v8_demo_uses_training_near_random_keyframe_sampling():
    sched = build_stage5_demo_scheduler_from_cfg(_train_v8_cfg(), _TinyDataset(), device=torch.device("cpu"))
    batch = sched.materialize_current_batch_without_advance()
    rm = batch["request_meta"]
    source_frame = int(batch["_scheduler_v8_aligned_info"]["source_frame_idx"])
    near = [int(x) for x in rm["near_random_frame_indices"]]
    assert len(near) == 2
    assert source_frame not in near
    assert {x // 3 for x in near} == {source_frame // 3}
    assert rm["target_frame_roles"] == ["source", "near_random", "near_random"]
    assert rm["target_image_roles"].count("near_random") == 4


def test_stage5_6_train_v8_demo_next_step_consumes_train_scheduler_batch():
    cfg = _train_v8_cfg(steps_per_block=1, switch=1)
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
    assert trainer.seen["runtime_policy"].reset_node_state_after_block is False
    assert stats["demo_scheduler_type"] == "train_v8_stage5_6"
    assert stats["target_frame_roles"][0] == "source"


def test_stage5_4_train_v8_demo_next_step_consumes_train_scheduler_batch():
    cfg = _train_v8_cfg(steps_per_block=1, switch=1, stage="5_4", scheduler_type="train_v8_stage5_4")
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    trainer = _Trainer()
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=trainer,
        device=torch.device("cpu"),
        stage="5_4",
    )
    controller.prime()
    stats = controller.step_current_block_once()
    assert trainer.seen["scheduler_node_sync"]["segment_local_step"] == 1
    assert trainer.seen["runtime_policy"].reset_node_state_after_block is False
    assert stats["demo_scheduler_type"] == "train_v8_stage5_4"
    assert stats["target_frame_roles"][0] == "source"


def test_stage5_6_train_v8_demo_runtime_policy_resets_only_on_scheduler_reset_event():
    cfg = _train_v8_cfg(steps_per_block=1, switch=1)
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
    controller._run_stage5_6_train_scheduler_step(
        minimal={"scene_id": 10, "segment_id": 0},
        scheduler_info={"segment_local_step": 1, "U": 1, "block_order": "step_major"},
        events=[{"type": "episode_end"}],
        defer_node_state_reset=False,
    )
    assert trainer.seen["runtime_policy"].reset_node_state_after_block is True


def test_stage5_6_demo_viewer_defaults_to_auto_rasterization():
    cfg = _train_v8_cfg()
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=_Trainer(),
        device=torch.device("cpu"),
        stage="5_6",
    )
    assert controller.viewer_rasterize_mode == "auto"


def test_stage5_6_demo_viewer_accepts_classic_rasterization_override():
    cfg = _train_v8_cfg()
    cfg.demo.viewer = {"rasterize_mode": "classic"}
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=_Trainer(),
        device=torch.device("cpu"),
        stage="5_6",
    )
    assert controller.viewer_rasterize_mode == "classic"


def test_stage5_demo_forward_render_cache_is_opt_in_and_captures_forward_params():
    cfg = _train_v8_cfg()
    cfg.demo.viewer = {"use_forward_render_cache": True}
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    trainer = _ForwardRenderTrainer()
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=trainer,
        device=torch.device("cpu"),
        stage="5_6",
    )
    controller.prime()
    cache = controller._display_render_cache
    assert isinstance(cache, dict)
    assert cache["key"] == (10, 0)
    assert cache["render_bg"]["means_r"].shape == (1, 3)
    assert trainer.seen["forward_batch_key"] == (10, 0)


def test_stage5_6_demo_render_initializes_node_state_from_primed_batch():
    cfg = _train_v8_cfg()
    sched = build_stage5_demo_scheduler_from_cfg(cfg, _TinyDataset(), device=torch.device("cpu"))
    trainer = _Trainer()
    trainer.node_states_bg = {}
    trainer.node_states_distant = {}
    trainer.node_states_rigid = {}

    def _init_states(batch):
        trainer.seen["init_batch_key"] = (int(batch["scene_id"]), int(batch["segment_id"]))
        trainer.node_states_bg[trainer.seen["init_batch_key"]] = object()
        trainer.node_states_distant[trainer.seen["init_batch_key"]] = object()
        trainer.node_states_rigid[trainer.seen["init_batch_key"]] = object()

    trainer._get_or_init_node_states_bg_rigid_distant = _init_states
    controller = Stage5DemoController(
        cfg=cfg,
        dataset=_TinyDataset(),
        scheduler=sched,
        trainer=trainer,
        device=torch.device("cpu"),
        stage="5_6",
    )
    controller.prime()
    controller._ensure_render_node_state_initialized((10, 0))
    assert trainer.seen["init_batch_key"] == (10, 0)


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


class _ForwardRenderTrainer(_Trainer):
    def forward(self, batch):
        self.seen["forward_batch_key"] = (int(batch["scene_id"]), int(batch["segment_id"]))
        return {
            "render_params": {
                "means_r": torch.zeros((1, 3), dtype=torch.float32),
                "scales_r": torch.ones((1, 3), dtype=torch.float32),
                "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "opacities_r": torch.ones((1,), dtype=torch.float32),
                "colors_r": torch.zeros((1, 1, 3), dtype=torch.float32),
            }
        }


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
