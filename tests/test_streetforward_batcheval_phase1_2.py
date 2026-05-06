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
from streetforward_eval.protocols import TestProtocolSpec, protocol_from_dict
from streetforward_eval.runner import RunnerRuntimeConfig, StreetForwardBatchEvalRunner
from streetforward_eval.snapshot_writer import SnapshotWriter


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
        self.render_calls = 0
        self.history_calls = 0
        self.reset_node_state_calls = 0
        self.last_train_sync: Dict[str, Any] | None = None
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
        return {"loss": 0.0, "num_targets": 1, "num_source_views": 3}

    def train_step(
        self,
        batch: Dict[str, Any],
        step: int | None = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = (batch, step, profile_phase_timing, sync_cuda_timing)
        self.train_calls += 1
        self.last_train_sync = dict(scheduler_node_sync or {})
        return {"loss": torch.tensor(0.0)}

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


def test_protocol_from_dict_accepts_omegaconf_nodes() -> None:
    cfg = OmegaConf.create(
        {
            "name": "exp2_storm20_sparse4",
            "sequence_length": 20,
            "input_offsets": [0, 5, 10, 15],
            "eval_offsets": "all",
            "steps_per_input": 8,
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
    assert float(rows[0]["psnr"]) > 30.0
    assert float(rows[0]["psnr_non_sky"]) > 30.0
