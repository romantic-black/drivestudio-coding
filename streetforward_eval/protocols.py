from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True)
class TestProtocolSpec:
    name: str
    data_mode: str

    sequence_length: int
    input_offsets: List[int]
    eval_offsets: List[int] | Literal["all"]

    camera_ids: List[int]
    camera_names: List[str]

    steps_per_input: int
    save_pre_update: bool
    save_each_iter_views: bool

    metric_primary_mask: str
    report_full_image: bool

    # Global optimization iterations at which the current state is rendered and
    # evaluated.  None preserves the legacy behavior (every iteration when
    # save_each_iter_views=true, otherwise only the final iteration).
    report_iterations: Optional[List[int]] = None

    input_count_label: Optional[str] = None
    train_block_size_label: Optional[str] = None


def resolve_eval_offsets(eval_offsets_any: Any, *, sequence_length: int) -> List[int]:
    if eval_offsets_any == "all":
        return list(range(int(sequence_length)))
    if isinstance(eval_offsets_any, ListConfig):
        eval_offsets_any = list(eval_offsets_any)
    if not isinstance(eval_offsets_any, list):
        raise ValueError(
            f"eval_offsets must be 'all' or list[int], got {type(eval_offsets_any).__name__}"
        )
    return [int(x) for x in eval_offsets_any]


def _as_list(value: Any, name: str) -> List[Any]:
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{name} must be list-like, got {type(value).__name__}")


def _as_mapping(value: Any, name: str) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{name} must be mapping-like, got {type(value).__name__}")


def validate_protocol(
    protocol: TestProtocolSpec,
    *,
    require_20frame_sparse4: bool = False,
) -> None:
    if str(protocol.data_mode) != "segment_finetune_train":
        raise ValueError(
            "batch_eval protocol requires data_mode='segment_finetune_train', "
            f"got {protocol.data_mode!r}"
        )
    if int(protocol.sequence_length) < 1:
        raise ValueError("sequence_length must be >= 1")
    if int(protocol.steps_per_input) < 1:
        raise ValueError("steps_per_input must be >= 1")
    if len(protocol.input_offsets) < 1:
        raise ValueError("input_offsets must not be empty")
    if len(protocol.camera_ids) < 1:
        raise ValueError("camera_ids must not be empty")
    if len(protocol.camera_names) != len(protocol.camera_ids):
        raise ValueError(
            "camera_names and camera_ids length mismatch: "
            f"{len(protocol.camera_names)} vs {len(protocol.camera_ids)}"
        )
    for cam_id in protocol.camera_ids:
        if int(cam_id) < 0:
            raise ValueError(f"camera id must be >= 0, got {cam_id}")

    for off in protocol.input_offsets:
        o = int(off)
        if o < 0 or o >= int(protocol.sequence_length):
            raise ValueError(
                f"input offset out of range: {o}, sequence_length={protocol.sequence_length}"
            )

    eval_offsets = resolve_eval_offsets(protocol.eval_offsets, sequence_length=int(protocol.sequence_length))
    if len(eval_offsets) < 1:
        raise ValueError("eval_offsets resolved to empty list")
    for off in eval_offsets:
        o = int(off)
        if o < 0 or o >= int(protocol.sequence_length):
            raise ValueError(
                f"eval offset out of range: {o}, sequence_length={protocol.sequence_length}"
            )

    if protocol.report_iterations is not None:
        report_iterations = [int(x) for x in protocol.report_iterations]
        if len(report_iterations) == 0:
            raise ValueError("report_iterations must not be empty when provided")
        if report_iterations != sorted(set(report_iterations)):
            raise ValueError("report_iterations must be strictly increasing and unique")
        total_iterations = int(len(protocol.input_offsets) * protocol.steps_per_input)
        for iteration in report_iterations:
            if iteration < 1 or iteration > total_iterations:
                raise ValueError(
                    "report_iterations values must be within the experiment optimization budget: "
                    f"iteration={iteration}, total_iterations={total_iterations}"
                )


def protocol_from_dict(
    *,
    exp_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> TestProtocolSpec:
    exp_cfg = _as_mapping(exp_cfg, "experiment config")
    global_cfg = _as_mapping(global_cfg, "batch_eval config")
    name = str(exp_cfg.get("name", "")).strip()
    if not name:
        raise ValueError("batch_eval.experiments[].name is required")

    data_mode = str(global_cfg.get("data_mode", "")).strip()
    if not data_mode:
        raise ValueError("batch_eval.data_mode is required")

    seq_len_any = exp_cfg.get("sequence_length")
    if seq_len_any is None:
        raise ValueError(f"experiment {name}: sequence_length is required")
    sequence_length = int(seq_len_any)

    input_offsets_any = exp_cfg.get("input_offsets")
    input_offsets_list = _as_list(input_offsets_any, f"experiment {name}: input_offsets")
    if len(input_offsets_list) == 0:
        raise ValueError(f"experiment {name}: input_offsets must be non-empty list[int]")
    input_offsets = [int(x) for x in input_offsets_list]

    if "eval_offsets" not in exp_cfg:
        raise ValueError(f"experiment {name}: eval_offsets is required")
    eval_offsets_any = exp_cfg["eval_offsets"]
    eval_offsets: List[int] | Literal["all"]
    if eval_offsets_any == "all":
        eval_offsets = "all"
    else:
        eval_offsets_list = _as_list(eval_offsets_any, f"experiment {name}: eval_offsets")
        eval_offsets = [int(x) for x in eval_offsets_list]

    steps_any = exp_cfg.get("steps_per_input")
    if steps_any is None:
        raise ValueError(f"experiment {name}: steps_per_input is required")
    steps_per_input = int(steps_any)

    report_iterations_any = exp_cfg.get("report_iterations")
    report_iterations = (
        None
        if report_iterations_any is None
        else [
            int(x)
            for x in _as_list(
                report_iterations_any,
                f"experiment {name}: report_iterations",
            )
        ]
    )

    cameras_cfg = global_cfg.get("cameras")
    cameras_cfg = _as_mapping(cameras_cfg, "batch_eval.cameras")
    camera_ids_any = cameras_cfg.get("ids")
    camera_names_any = cameras_cfg.get("names")
    camera_ids_list = _as_list(camera_ids_any, "batch_eval.cameras.ids")
    camera_names_list = _as_list(camera_names_any, "batch_eval.cameras.names")
    if len(camera_ids_list) == 0:
        raise ValueError("batch_eval.cameras.ids must be non-empty list[int]")
    if len(camera_names_list) == 0:
        raise ValueError("batch_eval.cameras.names must be non-empty list[str]")
    camera_ids = [int(x) for x in camera_ids_list]
    camera_names = [str(x) for x in camera_names_list]

    render_cfg = global_cfg.get("render", {}) or {}
    metrics_cfg = global_cfg.get("metrics", {}) or {}
    protocol = TestProtocolSpec(
        name=name,
        data_mode=data_mode,
        sequence_length=sequence_length,
        input_offsets=input_offsets,
        eval_offsets=eval_offsets,
        camera_ids=camera_ids,
        camera_names=camera_names,
        steps_per_input=steps_per_input,
        save_pre_update=bool(render_cfg.get("save_pre_update", True)),
        save_each_iter_views=bool(render_cfg.get("save_each_iter_views", True)),
        metric_primary_mask=str(metrics_cfg.get("primary_mask", "non_sky_non_ego")),
        report_full_image=bool(metrics_cfg.get("report_full_image", True)),
        report_iterations=report_iterations,
        input_count_label=(
            None
            if exp_cfg.get("input_count_label") is None
            else str(exp_cfg.get("input_count_label"))
        ),
        train_block_size_label=(
            None
            if exp_cfg.get("train_block_size_label") is None
            else str(exp_cfg.get("train_block_size_label"))
        ),
    )
    return protocol
