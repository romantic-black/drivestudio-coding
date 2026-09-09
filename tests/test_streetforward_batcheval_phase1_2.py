from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from omegaconf import OmegaConf

from streetforward_eval.batch_builder import validate_update_target_refs
from streetforward_eval.episode_builder import TestEpisodeSpec, build_test_episode_specs
from streetforward_eval.metrics import MetricAccumulator
from streetforward_eval.protocols import TestProtocolSpec, protocol_from_dict, validate_protocol
from streetforward_eval.runner import RunnerRuntimeConfig, StreetForwardBatchEvalRunner
from streetforward_eval.snapshot_writer import SnapshotWriter
from streetforward_eval.summary import build_optimization_curve_rows, build_summary_rows


@dataclass
class _DummySegmentIndex:
    frame_indices: List[int]
    train_frame_set: set[int]
    num_cams: int


class _DummyDataset:
    def list_segment_ids(self, scene_id: int) -> List[int]:
        assert int(scene_id) == 1
        return [2]

    def get_segment_index(self, scene_id: int, segment_id: int) -> _DummySegmentIndex:
        assert int(scene_id) == 1
        assert int(segment_id) == 2
        frames = list(range(100, 140))
        return _DummySegmentIndex(
            frame_indices=frames,
            train_frame_set=set(frames),
            num_cams=3,
        )


@dataclass
class _DummyV7SegmentIndex:
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    num_cams: int


class _DummyV7Dataset:
    def list_segment_ids(self, scene_id: int) -> List[int]:
        assert int(scene_id) == 1
        return [2]

    def get_segment_index(self, scene_id: int, segment_id: int) -> _DummyV7SegmentIndex:
        assert int(scene_id) == 1
        assert int(segment_id) == 2
        keyframes = list(range(12))
        return _DummyV7SegmentIndex(
            keyframe_indices=keyframes,
            keyframe_to_frames={int(k): [100 + int(k)] for k in keyframes},
            num_cams=1,
        )


def _make_protocol_exp2() -> TestProtocolSpec:
    return TestProtocolSpec(
        name="exp2_storm20_sparse4",
        data_mode="segment_finetune_train",
        sequence_length=20,
        input_offsets=[0, 5, 10, 15],
        eval_offsets="all",
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        steps_per_input=2,
        save_pre_update=True,
        save_each_iter_views=True,
        metric_primary_mask="full_image",
        report_full_image=True,
    )


def test_episode_builder_relative_to_absolute_mapping() -> None:
    dataset = _DummyDataset()
    protocol = _make_protocol_exp2()
    specs = build_test_episode_specs(
        dataset=dataset,
        scene_ids=[1],
        protocol=protocol,
        segment_policy="all",
        window_policy="sliding",
        stride=20,
        require_full_window=True,
        max_episodes_per_scene=1,
        max_total_episodes=1,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.frame_offsets == list(range(20))
    assert spec.frame_ids == list(range(100, 120))
    assert spec.input_offsets == [0, 5, 10, 15]
    assert spec.input_frame_ids == [100, 105, 110, 115]
    assert spec.eval_offsets == list(range(20))
    assert spec.eval_frame_ids == list(range(100, 120))


def test_scheduler_v7_episode_builder_allows_non_strict_protocol_window() -> None:
    from tools.eval_streetforward_benchmark import _build_scheduler_v7_episode_specs

    cfg = OmegaConf.create(
        {
            "scheduler_v7": {
                "enable": True,
                "episode": {
                    "blocks_per_episode": 3,
                    "total_target_frames": 3,
                },
            },
            "batch_eval": {
                "dataset": {
                    "scene_ids": [1],
                },
            },
        }
    )
    protocol = TestProtocolSpec(
        name="exp_v7_sparse",
        data_mode="segment_finetune_train",
        sequence_length=10,
        input_offsets=[0, 2, 4],
        eval_offsets="all",
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )

    specs = _build_scheduler_v7_episode_specs(
        cfg=cfg,
        dataset=_DummyV7Dataset(),
        protocol=protocol,
        max_total_episodes=None,
    )

    assert len(specs) == 1
    spec = specs[0]
    assert len(spec.frame_ids) == 10
    assert spec.input_offsets == [0, 2, 4]
    assert spec.eval_offsets == list(range(10))
    assert spec.input_frame_ids == [102, 104, 106]
    assert spec.eval_frame_ids == list(range(102, 112))


def test_scheduler_v7_episode_builder_honors_batch_eval_start_at() -> None:
    from tools.eval_streetforward_benchmark import _build_scheduler_v7_episode_specs

    cfg = OmegaConf.create(
        {
            "scheduler_v7": {
                "enable": True,
                "episode": {
                    "blocks_per_episode": 3,
                    "total_target_frames": 3,
                },
            },
            "batch_eval": {
                "dataset": {
                    "scene_ids": [1],
                    "start_at": {"scene_id": 1, "segment_id": 2, "frame_id": 100},
                },
            },
        }
    )
    protocol = TestProtocolSpec(
        name="exp_v7_sparse",
        data_mode="segment_finetune_train",
        sequence_length=10,
        input_offsets=[0, 2, 4],
        eval_offsets="all",
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )

    specs = _build_scheduler_v7_episode_specs(
        cfg=cfg,
        dataset=_DummyV7Dataset(),
        protocol=protocol,
        max_total_episodes=None,
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.segment_id == 2
    assert spec.sequence_start_pos == 0
    assert spec.frame_ids[0] == 100
    assert spec.input_frame_ids == [100, 102, 104]


class _StartAtDataset:
    def list_segment_ids(self, scene_id: int) -> List[int]:
        assert int(scene_id) == 10
        return [0, 1, 2]

    def get_segment_index(self, scene_id: int, segment_id: int) -> _DummySegmentIndex:
        assert int(scene_id) == 10
        starts = {0: 0, 1: 73, 2: 200}
        start = starts[int(segment_id)]
        frames = list(range(start, start + 60))
        return _DummySegmentIndex(
            frame_indices=frames,
            train_frame_set=set(frames),
            num_cams=3,
        )


def test_episode_builder_start_at_frame_id_rebases_sliding_starts() -> None:
    protocol = TestProtocolSpec(
        name="exp_start_at",
        data_mode="segment_finetune_train",
        sequence_length=10,
        input_offsets=[1, 3, 5, 7, 9],
        eval_offsets="all",
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    specs = build_test_episode_specs(
        dataset=_StartAtDataset(),
        scene_ids=[10],
        protocol=protocol,
        segment_policy="all",
        window_policy="sliding",
        stride=30,
        require_full_window=True,
        max_episodes_per_scene=None,
        max_total_episodes=None,
        start_at={"scene_id": 10, "segment_id": 1, "frame_id": 100},
    )
    assert [(s.scene_id, s.segment_id, s.sequence_start_pos, s.frame_ids[0]) for s in specs] == [
        (10, 1, 27, 100),
        (10, 2, 0, 200),
        (10, 2, 30, 230),
    ]
    assert specs[0].input_frame_ids == [101, 103, 105, 107, 109]
    assert specs[0].episode_uid == "scene010_seg001_start000027"


def test_update_targets_reject_unobserved_frames() -> None:
    with pytest.raises(ValueError):
        validate_update_target_refs(
            update_target_image_refs=[(100, 0), (120, 1)],
            observed_frame_ids=[100, 105],
            camera_ids=[0, 1, 2],
        )


class _FakeModel:
    def __init__(self) -> None:
        self.update_calls = 0
        self.train_calls = 0
        self.infer_train_batch_calls = 0
        self.render_calls = 0
        self.history_calls = 0
        self.reset_node_state_calls = 0
        self.last_train_sync: Dict[str, Any] | None = None
        self.last_infer_policy: Any = None
        self.infer_batches: List[Dict[str, Any]] = []
        self.update_batches: List[Dict[str, Any]] = []
        self.last_render_refs: List[Tuple[int, int]] = []

    def reset_for_segment_eval(self, batch: Dict[str, Any]) -> None:
        _ = batch

    def eval_sparse_update_step(
        self,
        batch: Dict[str, Any],
        *,
        local_iter: int,
        num_local_iters: int,
        amp: bool = True,
        update_node_state: bool = True,
        update_hidden_state: bool = True,
        update_view_transient: bool = True,
        update_step_norm_ema: bool = True,
    ) -> Dict[str, Any]:
        _ = (
            batch,
            local_iter,
            num_local_iters,
            amp,
            update_node_state,
            update_hidden_state,
            update_view_transient,
            update_step_norm_ema,
        )
        self.update_calls += 1
        self.update_batches.append(batch)
        return {"loss": 0.0, "num_targets": 1, "num_source_views": 3}

    def train_step(
        self,
        batch: Dict[str, Any],
        step: int | None = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Dict[str, Any] | None = None,
        runtime_policy: Any = None,
    ) -> Dict[str, Any]:
        _ = (batch, step, profile_phase_timing, sync_cuda_timing, runtime_policy)
        self.train_calls += 1
        self.last_train_sync = dict(scheduler_node_sync or {})
        return {"loss": torch.tensor(0.0)}

    def inference_step_from_train_batch(
        self,
        batch: Dict[str, Any],
        step: int | None = None,
        scheduler_node_sync: Dict[str, Any] | None = None,
        runtime_policy: Any = None,
    ) -> Dict[str, Any]:
        _ = (step, scheduler_node_sync)
        self.infer_train_batch_calls += 1
        self.last_infer_policy = runtime_policy
        self.infer_batches.append(batch)
        return {"loss": 0.0, "pred_rgbs": [], "gt_images": []}

    def _render_scene_views_from_current_state(
        self,
        batch: Dict[str, Any],
        render_items: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = batch
        n = len(render_items)
        return torch.zeros((n, 4, 4, 3)), torch.ones((n, 4, 4, 1))

    def eval_sparse_render_frames(
        self,
        *,
        scene_id: int,
        segment_id: int,
        image_refs: List[Tuple[int, int]],
        camera_ids: List[int],
        save_dir: Path | None = None,
        amp: bool = True,
    ) -> Dict[str, Any]:
        _ = (scene_id, segment_id, camera_ids, save_dir, amp)
        self.render_calls += 1
        self.last_render_refs = [(int(f), int(c)) for f, c in image_refs]
        rows: List[Dict[str, Any]] = []
        for fid, cam in image_refs:
            img = torch.full((4, 4, 3), 0.5, dtype=torch.float32)
            rows.append(
                {
                    "frame_idx": int(fid),
                    "cam_idx": int(cam),
                    "pred_rgb": img,
                    "gt_image": img.clone(),
                    "sky_mask": torch.zeros((4, 4), dtype=torch.float32),
                }
            )
        return {"rows": rows}

    def eval_sparse_record_history(self, batch: Dict[str, Any]) -> None:
        _ = batch
        self.history_calls += 1

    def reset_node_state(self) -> None:
        self.reset_node_state_calls += 1


def test_runner_iter_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        return {"dummy": True, "kwargs": kwargs}

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)

    protocol = TestProtocolSpec(
        name="exp1_single_frame",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        steps_per_input=3,
        save_pre_update=True,
        save_each_iter_views=True,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp1_single_frame",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        input_image_refs=[(100, 0), (100, 1), (100, 2)],
        eval_image_refs=[(100, 0), (100, 1), (100, 2)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    metric_acc = MetricAccumulator(
        output_dir=tmp_path / "metrics",
        protocol=protocol,
        min_valid_pixels=1,
        compute_ssim=False,
        compute_lpips=False,
    )
    writer = SnapshotWriter(output_dir=tmp_path / "snapshots")
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=writer,
        metric_acc=metric_acc,
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(history_record_on_input_exit=True),
    )
    out = runner.run_episode(spec)
    assert model.update_calls == 3
    assert model.history_calls == 1
    assert model.render_calls == 4  # pre + 3 update renders
    assert out["final_iter"] == 3


def test_runner_renders_only_requested_optimization_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from streetforward_eval import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build_update_batch_from_refs",
        lambda **kwargs: {"dummy": True, "kwargs": kwargs},
    )
    protocol = TestProtocolSpec(
        name="single_frame_curve",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0, 1, 2],
        camera_names=["front", "front_left", "front_right"],
        steps_per_input=8,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
        report_iterations=[1, 2, 4, 8],
    )
    spec = TestEpisodeSpec(
        exp_name=protocol.name,
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0, 1, 2],
        camera_names=["front", "front_left", "front_right"],
        input_image_refs=[(100, 0), (100, 1), (100, 2)],
        eval_image_refs=[(100, 0), (100, 1), (100, 2)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    metric_acc = MetricAccumulator(
        output_dir=tmp_path / "metrics",
        protocol=protocol,
        min_valid_pixels=1,
        compute_ssim=False,
        compute_lpips=False,
    )
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=metric_acc,
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(history_record_on_input_exit=True),
    )

    runner.run_episode(spec)

    assert model.update_calls == 8
    assert model.render_calls == 4
    assert sorted({int(row["global_iter"]) for row in metric_acc.iter_rows}) == [1, 2, 4, 8]
    curve = build_optimization_curve_rows(metric_acc.iter_rows)
    assert [row["optimization_steps"] for row in curve] == [1, 2, 4, 8]
    assert all(row["num_views"] == 3 for row in curve)


def test_protocol_from_dict_accepts_omegaconf_nodes() -> None:
    cfg = OmegaConf.create(
        {
            "name": "exp2_storm20_sparse4",
            "sequence_length": 20,
            "input_offsets": [0, 5, 10, 15],
            "eval_offsets": "all",
            "steps_per_input": 8,
            "report_iterations": [1, 2, 4, 8],
        }
    )
    global_cfg = OmegaConf.create(
        {
            "data_mode": "segment_finetune_train",
            "cameras": {"ids": [0, 1, 2], "names": ["front_left", "front", "front_right"]},
            "render": {"save_pre_update": True, "save_each_iter_views": True},
            "metrics": {"primary_mask": "non_sky_non_ego", "report_full_image": True},
        }
    )
    protocol = protocol_from_dict(exp_cfg=cfg, global_cfg=global_cfg)
    assert protocol.sequence_length == 20
    assert protocol.input_offsets == [0, 5, 10, 15]
    assert protocol.eval_offsets == "all"
    assert protocol.report_iterations == [1, 2, 4, 8]


def test_load_cfg_supports_historical_git_base(tmp_path: Path) -> None:
    from tools.eval_streetforward_benchmark import load_cfg

    config_path = tmp_path / "historical_overlay.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base_config_file: configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml",
                'base_config_git_revision: "59266ef"',
                "batch_eval:",
                "  enable: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_cfg(str(config_path))

    assert str(cfg.model.stage) == "5_3"
    assert int(cfg.model.struct_decoder.feat_2d_channels) == 32
    assert float(cfg.model.update_gate.min_gate) == pytest.approx(0.05)
    assert bool(cfg.batch_eval.enable) is True


def test_protocol_rejects_report_iteration_outside_budget() -> None:
    protocol = TestProtocolSpec(
        name="single_frame_curve",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=8,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
        report_iterations=[1, 2, 16],
    )

    with pytest.raises(ValueError, match="optimization budget"):
        validate_protocol(protocol)


def test_runner_update_targets_source_frame_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    captured_targets: List[List[Tuple[int, int]]] = []

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        captured_targets.append(list(kwargs["update_target_image_refs"]))
        return {"dummy": True}

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)
    protocol = TestProtocolSpec(
        name="exp2_storm20_sparse4",
        data_mode="segment_finetune_train",
        sequence_length=20,
        input_offsets=[0, 5],
        eval_offsets=[0, 5],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp2_storm20_sparse4",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=list(range(20)),
        frame_ids=list(range(100, 120)),
        input_offsets=[0, 5],
        eval_offsets=[0, 5],
        input_frame_ids=[100, 105],
        eval_frame_ids=[100, 105],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        input_image_refs=[(100, 0), (100, 1), (100, 2), (105, 0), (105, 1), (105, 2)],
        eval_image_refs=[(100, 0), (100, 1), (100, 2), (105, 0), (105, 1), (105, 2)],
        episode_uid="scene001_seg002_start000000",
    )
    runner = StreetForwardBatchEvalRunner(
        model=_FakeModel(),
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
        ),
    )
    runner.run_episode(spec)
    assert len(captured_targets) == 2
    assert [x[0] for x in captured_targets[0][:3]] == [100, 100, 100]
    assert [x[0] for x in captured_targets[1][:3]] == [105, 105, 105]
    assert set([x[0] for x in captured_targets[1]]) == {100, 105}


def test_runner_renders_final_even_when_each_iter_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {"dummy": True}

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)
    protocol = TestProtocolSpec(
        name="exp1_single_frame",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        steps_per_input=3,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp1_single_frame",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0, 1, 2],
        camera_names=["front_left", "front", "front_right"],
        input_image_refs=[(100, 0), (100, 1), (100, 2)],
        eval_image_refs=[(100, 0), (100, 1), (100, 2)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(history_record_on_input_exit=False),
    )
    out = runner.run_episode(spec)
    assert model.update_calls == 3
    assert model.render_calls == 1
    assert out["final_iter"] == 3


def test_runner_step_major_order_and_visited_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    captured_sources: List[List[Tuple[int, int]]] = []
    captured_targets: List[List[Tuple[int, int]]] = []

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        captured_sources.append(list(kwargs["source_image_refs"]))
        captured_targets.append(list(kwargs["update_target_image_refs"]))
        return {"dummy": True}

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)
    protocol = TestProtocolSpec(
        name="exp_step_major",
        data_mode="segment_finetune_train",
        sequence_length=3,
        input_offsets=[0, 1, 2],
        eval_offsets=[0, 1, 2],
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=2,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp_step_major",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0, 1, 2],
        frame_ids=[100, 105, 110],
        input_offsets=[0, 1, 2],
        eval_offsets=[0, 1, 2],
        input_frame_ids=[100, 105, 110],
        eval_frame_ids=[100, 105, 110],
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(100, 0), (105, 0), (110, 0)],
        eval_image_refs=[(100, 0), (105, 0), (110, 0)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
            block_order="step_major",
            step_major_switch_interval_steps=1,
            target_frame_policy="visited_episode_frames",
            max_target_frames_including_source=2,
        ),
    )
    runner.run_episode(spec)
    assert [refs[0][0] for refs in captured_sources] == [100, 105, 110, 100, 105, 110]
    assert [[ref[0] for ref in refs] for refs in captured_targets[:4]] == [
        [100],
        [105, 100],
        [110, 105],
        [100, 105],
    ]
    assert model.update_calls == 6
    assert all(batch["_scheduler_v8_aligned_info"]["scheduler_version"] == "v8" for batch in model.update_batches)


class _Stage56Dataset:
    def __init__(self) -> None:
        self.raw_batches: List[Dict[str, Any]] = []

    def _assemble_segment_batch_from_image_refs(
        self,
        scene_id: int,
        segment_id: int,
        source_image_refs: List[Tuple[int, int]],
        target_image_refs: List[Tuple[int, int]],
        aux_image_refs: List[Tuple[int, int]] | None = None,
        *,
        include_test: bool,
        test_image_refs: Any,
        enforce_target0_equals_source: bool,
        target_ref_purpose: str = "train",
    ) -> Dict[str, Any]:
        _ = (aux_image_refs, include_test, test_image_refs, enforce_target0_equals_source, target_ref_purpose)
        raw = {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "source_image_refs": list(source_image_refs),
            "target_image_refs": list(target_image_refs),
            "request_meta": {
                "source_image_refs": list(source_image_refs),
                "target_image_refs": list(target_image_refs),
            },
        }
        self.raw_batches.append(raw)
        return raw


def _fake_minimal_from_raw(raw: Dict[str, Any], device: torch.device, num_targets: int, include_source_for_2d: bool, **_: Any) -> Dict[str, Any]:
    _ = include_source_for_2d
    target_refs = list(raw.get("request_meta", {}).get("target_image_refs") or raw.get("target_image_refs") or [])
    source_refs = list(raw.get("request_meta", {}).get("source_image_refs") or raw.get("source_image_refs") or [])
    targets: List[Dict[str, Any]] = []
    for frame_idx, cam_idx in target_refs[:num_targets]:
        targets.append(
            {
                "frame_idx": int(frame_idx),
                "cam_idx": int(cam_idx),
                "view": object(),
                "gt_image": torch.zeros((4, 4, 3), device=device),
                "sky_mask": torch.zeros((4, 4), device=device),
            }
        )
    out = {
        "scene_id": int(raw["scene_id"]),
        "segment_id": int(raw["segment_id"]),
        "request_meta": dict(raw.get("request_meta") or {}),
        "targets": targets,
        "source_views": [object() for _ in source_refs],
        "source_images": [torch.zeros((4, 4, 3), device=device) for _ in source_refs],
        "source_frame_idx": int(source_refs[0][0]) if source_refs else -1,
    }
    for key in ("_scheduler_v4_aligned_info", "_scheduler_v7_aligned_info", "_scheduler_v8_aligned_info"):
        if key in raw:
            out[key] = dict(raw[key])
    return out


def test_stage5_6_adjacent_nearby_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod
    from streetforward_eval import stage5_6_runtime as stage56_mod

    monkeypatch.setattr(runner_mod, "convert_batch_to_minimal_format", _fake_minimal_from_raw)
    monkeypatch.setattr(stage56_mod, "convert_batch_to_minimal_format", _fake_minimal_from_raw)
    protocol = TestProtocolSpec(
        name="exp_stage56_nearby",
        data_mode="segment_finetune_train",
        sequence_length=10,
        input_offsets=[1, 3, 5, 7, 9],
        eval_offsets="all",
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name=protocol.name,
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=list(range(10)),
        frame_ids=list(range(100, 110)),
        input_offsets=[1, 3, 5, 7, 9],
        eval_offsets=list(range(10)),
        input_frame_ids=[101, 103, 105, 107, 109],
        eval_frame_ids=list(range(100, 110)),
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(101, 0), (103, 0), (105, 0), (107, 0), (109, 0)],
        eval_image_refs=[(f, 0) for f in range(100, 110)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=_Stage56Dataset(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            mode="inference_only",
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
            stage5_6_enable_nearby_feedback=True,
            update_camera_ids=[0],
            target_frame_policy="visited_episode_frames",
            max_target_frames_including_source=3,
        ),
    )
    runner.run_episode(spec)
    nearby = [
        list((b.get("request_meta") or {}).get("near_random_frame_indices") or [])
        for b in model.infer_batches
    ]
    assert nearby == [[100, 102], [102, 104], [104, 106], [106, 108], [108]]
    for batch in model.infer_batches:
        rm = batch["request_meta"]
        assert len(rm["target_image_refs"]) == len(rm["target_image_roles"])
        assert "near_random" in rm["target_image_roles"]
        assert batch["_scheduler_v8_aligned_info"]["scheduler_version"] == "v8"
        assert batch["_scheduler_v8_aligned_info"]["episode_idx_global"] == 0


class _FakeSkyBranch:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.reset_calls = 0

    def reset_runtime_state(self) -> None:
        self.reset_calls += 1

    def forward_scene_batch(self, batch: Dict[str, Any], scene_pack: Any, *, writeback: bool = False) -> Any:
        assert writeback is True
        assert scene_pack.source_rgb.shape[-1] == 3
        assert scene_pack.target_rgb.shape[-1] == 3
        _ = batch
        self.forward_calls += 1
        return object()


def test_stage5_6_sky_branch_updates_after_each_scene_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod
    from streetforward_eval import stage5_6_runtime as stage56_mod

    monkeypatch.setattr(runner_mod, "convert_batch_to_minimal_format", _fake_minimal_from_raw)
    monkeypatch.setattr(stage56_mod, "convert_batch_to_minimal_format", _fake_minimal_from_raw)
    protocol = TestProtocolSpec(
        name="exp_stage56_sky",
        data_mode="segment_finetune_train",
        sequence_length=2,
        input_offsets=[0, 1],
        eval_offsets=[0, 1],
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=2,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name=protocol.name,
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0, 1],
        frame_ids=[100, 101],
        input_offsets=[0, 1],
        eval_offsets=[0, 1],
        input_frame_ids=[100, 101],
        eval_frame_ids=[100, 101],
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(100, 0), (101, 0)],
        eval_image_refs=[(100, 0), (101, 0)],
        episode_uid="scene001_seg002_start000000",
    )
    sky = _FakeSkyBranch()
    runner = StreetForwardBatchEvalRunner(
        model=_FakeModel(),
        dataset=_Stage56Dataset(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            mode="inference_only",
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
            stage5_6_enable_nearby_feedback=True,
            update_camera_ids=[0],
            target_frame_policy="visited_episode_frames",
        ),
        sky_branch=sky,
    )
    runner.run_episode(spec)
    assert sky.reset_calls == 1
    assert sky.forward_calls == 4


def test_runner_no_grad_false_uses_train_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {"dummy": True}

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)
    protocol = TestProtocolSpec(
        name="exp_train_eval",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp_train_eval",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(100, 0)],
        eval_image_refs=[(100, 0)],
        episode_uid="scene001_seg002_start000000",
    )
    model = _FakeModel()
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            no_grad=False,
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
        ),
    )
    runner.run_episode(spec)
    assert model.update_calls == 0
    assert model.train_calls == 1
    assert model.last_train_sync == {"U": 1, "segment_local_step": 1, "reset_after_block": False}


def test_runner_scheduler_v7_block_window_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from streetforward_eval import runner as runner_mod

    captured_targets: List[List[Tuple[int, int]]] = []

    def _fake_build_update_batch_from_refs(**kwargs: Any) -> Dict[str, Any]:
        captured_targets.append(list(kwargs["update_target_image_refs"]))
        return {"dummy": True, "request_meta": dict(kwargs)}

    class _CaptureTrainModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.train_batches: List[Dict[str, Any]] = []

        def train_step(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
            self.train_batches.append(batch)
            return super().train_step(batch, **kwargs)

    monkeypatch.setattr(runner_mod, "build_update_batch_from_refs", _fake_build_update_batch_from_refs)
    protocol = TestProtocolSpec(
        name="exp_v7",
        data_mode="segment_finetune_train",
        sequence_length=5,
        input_offsets=[0, 1, 2],
        eval_offsets="all",
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="full_image",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp_v7",
        scene_id=1,
        segment_id=2,
        episode_idx=3,
        sequence_start_pos=4,
        frame_offsets=[0, 1, 2, 3, 4],
        frame_ids=[100, 101, 102, 103, 104],
        input_offsets=[0, 1, 2],
        eval_offsets=[0, 1, 2, 3, 4],
        input_frame_ids=[100, 101, 102],
        eval_frame_ids=[100, 101, 102, 103, 104],
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(100, 0), (101, 0), (102, 0)],
        eval_image_refs=[(100, 0), (101, 0), (102, 0), (103, 0), (104, 0)],
        episode_uid="scene001_seg002_kfstart000004",
    )
    model = _CaptureTrainModel()
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=object(),
        protocol=protocol,
        writer=SnapshotWriter(output_dir=tmp_path / "snapshots"),
        metric_acc=MetricAccumulator(
            output_dir=tmp_path / "metrics",
            protocol=protocol,
            min_valid_pixels=1,
            compute_ssim=False,
            compute_lpips=False,
        ),
        device=torch.device("cpu"),
        runtime_cfg=RunnerRuntimeConfig(
            mode="segment_finetune_train",
            no_grad=False,
            reset_state_per_episode=False,
            history_record_on_input_exit=False,
            target_frame_policy="scheduler_v7_block_window",
            max_target_frames_including_source=3,
        ),
    )
    runner.run_episode(spec)
    assert [[ref[0] for ref in refs] for refs in captured_targets] == [
        [100, 101, 102],
        [101, 102, 103],
        [102, 103, 104],
    ]
    assert [b["_scheduler_v7_aligned_info"]["scheduler_version"] for b in model.train_batches] == ["v7", "v7", "v7"]
    assert [b["_scheduler_v7_aligned_info"]["block_idx_global"] for b in model.train_batches] == [9, 10, 11]


def test_metrics_non_ego_mask_missing_does_not_crash(tmp_path: Path) -> None:
    protocol = TestProtocolSpec(
        name="exp1_single_frame",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0],
        camera_names=["front_left"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="non_sky_non_ego",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp1_single_frame",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0],
        camera_names=["front_left"],
        input_image_refs=[(100, 0)],
        eval_image_refs=[(100, 0)],
        episode_uid="scene001_seg002_start000000",
    )
    metric_acc = MetricAccumulator(
        output_dir=tmp_path / "metrics",
        protocol=protocol,
        min_valid_pixels=1,
        compute_ssim=False,
        compute_lpips=False,
    )
    img = torch.full((4, 4, 3), 0.5, dtype=torch.float32)
    render_rows: List[Dict[str, Any]] = [
        {
            "frame_idx": 100,
            "cam_idx": 0,
            "pred_rgb": img,
            "gt_image": img.clone(),
            "sky_mask": torch.zeros((4, 4), dtype=torch.float32),
            # egocar_mask intentionally missing.
        }
    ]
    rows = metric_acc.add_iteration_rows(
        spec=spec,
        global_iter=1,
        is_pre_update=False,
        input_index=0,
        input_frame_id=100,
        local_step=1,
        render_rows=render_rows,
    )
    assert len(rows) == 1
    assert rows[0]["metric_group"] == "reconstruction"
    assert float(rows[0]["psnr"]) > 30.0
    assert float(rows[0]["psnr_non_sky"]) > 30.0


def test_summary_splits_reconstruction_and_nvs() -> None:
    base: Dict[str, Any] = {
        "exp_name": "exp_split",
        "checkpoint": "",
        "variant": "",
        "input_count_label": "2",
        "train_block_size_label": "",
        "episode_uid": "episode_1",
        "cam_name": "front",
        "l1": 0.0,
        "ssim": 0.0,
        "ssim_non_sky": 0.0,
        "lpips": 0.0,
        "lpips_non_sky": 0.0,
    }
    rows = [
        {
            **base,
            "eval_frame_id": 100,
            "frame_group": "input",
            "is_input_frame": True,
            "metric_group": "reconstruction",
            "psnr": 10.0,
            "l1": 0.10,
            "ssim": 0.80,
            "lpips": 0.20,
        },
        {
            **base,
            "eval_frame_id": 101,
            "frame_group": "interp",
            "is_input_frame": False,
            "metric_group": "nvs",
            "psnr": 20.0,
            "l1": 0.20,
            "ssim": 0.90,
            "lpips": 0.30,
        },
    ]
    summary = build_summary_rows(rows)
    assert len(summary) == 1
    out = summary[0]
    assert out["num_views_reconstruction"] == 1
    assert out["num_views_nvs"] == 1
    assert out["mean_psnr_reconstruction"] == pytest.approx(10.0)
    assert out["mean_psnr_nvs"] == pytest.approx(20.0)
    assert out["mean_l1_reconstruction"] == pytest.approx(0.10)
    assert out["mean_l1_nvs"] == pytest.approx(0.20)
    assert out["mean_ssim_reconstruction"] == pytest.approx(0.80)
    assert out["mean_ssim_nvs"] == pytest.approx(0.90)
    assert out["mean_lpips_reconstruction"] == pytest.approx(0.20)
    assert out["mean_lpips_nvs"] == pytest.approx(0.30)


def test_metrics_compute_ssim(tmp_path: Path) -> None:
    protocol = TestProtocolSpec(
        name="exp1_single_frame",
        data_mode="segment_finetune_train",
        sequence_length=1,
        input_offsets=[0],
        eval_offsets=[0],
        camera_ids=[0],
        camera_names=["front"],
        steps_per_input=1,
        save_pre_update=False,
        save_each_iter_views=False,
        metric_primary_mask="non_sky",
        report_full_image=True,
    )
    spec = TestEpisodeSpec(
        exp_name="exp1_single_frame",
        scene_id=1,
        segment_id=2,
        episode_idx=0,
        sequence_start_pos=0,
        frame_offsets=[0],
        frame_ids=[100],
        input_offsets=[0],
        eval_offsets=[0],
        input_frame_ids=[100],
        eval_frame_ids=[100],
        camera_ids=[0],
        camera_names=["front"],
        input_image_refs=[(100, 0)],
        eval_image_refs=[(100, 0)],
        episode_uid="scene001_seg002_start000000",
    )
    metric_acc = MetricAccumulator(
        output_dir=tmp_path / "metrics",
        protocol=protocol,
        min_valid_pixels=1,
        compute_ssim=True,
        compute_lpips=False,
    )
    img = torch.rand((8, 8, 3), dtype=torch.float32)
    rows = metric_acc.add_iteration_rows(
        spec=spec,
        global_iter=1,
        is_pre_update=False,
        input_index=0,
        input_frame_id=100,
        local_step=1,
        render_rows=[
            {
                "frame_idx": 100,
                "cam_idx": 0,
                "pred_rgb": img,
                "gt_image": img.clone(),
                "sky_mask": torch.zeros((8, 8), dtype=torch.float32),
            }
        ],
    )
    assert len(rows) == 1
    assert float(rows[0]["ssim"]) > 0.99
    assert float(rows[0]["ssim_full"]) > 0.99
    assert float(rows[0]["ssim_non_sky"]) > 0.99
