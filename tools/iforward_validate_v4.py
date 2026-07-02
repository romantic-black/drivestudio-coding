from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf

from datasets.iforward_stage2_3.scheduler import Stage23Scheduler
from models.iforward.protocols.validation_recipes import build_validation_v4_plans
from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import TraceRecorder
from models.iforward.validation_v4.html_exporter import export_html_report, export_legacy_rows_html_report
from tools.train_iforward import build_iforward_trainer_from_cfg
from tools.train_iforward import _sequence10_minimal_from_scheduler_batch as _convert_batch
from tools.train_minimal_streetforward_stage4_3_iforward_common import build_multi_scene_dataset_v4


@dataclass
class RuntimeBundle:
    cfg: Any
    dataset: Any
    model: Any
    device: torch.device


def _parse_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_cfg(path: str, opts: Sequence[str]) -> Any:
    cfg = OmegaConf.load(path)
    if opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(opts)))
    return cfg


def _load_model_weights(model: Any, checkpoint: str) -> dict[str, Any]:
    if not checkpoint:
        return {}
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict payload, got {type(payload)}: {checkpoint}")
    state = payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"checkpoint missing model_state_dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    runtime_loader = getattr(model, "load_runtime_state_from_checkpoint", None)
    if callable(runtime_loader):
        runtime_loader(payload)
    return payload


def build_iforward_runtime_from_cfg(cfg: Any, *, checkpoint: str = "", device: str | None = None) -> RuntimeBundle:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = build_multi_scene_dataset_v4(cfg, device_obj)
    model = build_iforward_trainer_from_cfg(cfg, device_obj)
    _load_model_weights(model, checkpoint)
    if hasattr(model, "eval"):
        model.eval()
    return RuntimeBundle(cfg=cfg, dataset=dataset, model=model, device=device_obj)


def _make_scheduler(cfg: Any, dataset: Any) -> Stage23Scheduler:
    sched_cfg = cfg.get("scheduler_stage3_2", None) if hasattr(cfg, "get") else None
    if sched_cfg is None or not bool(sched_cfg.get("enable", False)):
        sched_cfg = cfg.get("scheduler_stage3_0", None) if hasattr(cfg, "get") else None
    if sched_cfg is None or not bool(sched_cfg.get("enable", False)):
        sched_cfg = cfg.get("scheduler_v3", {}) if hasattr(cfg, "get") else {}
    producer_cfg = dict((sched_cfg or {}).get("producer", {}) or {})
    producer_cfg["enable"] = False
    return Stage23Scheduler(dataset=dataset, cfg=cfg, producer_cfg=producer_cfg, fail_fast=False)


def _read_legacy_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "iforward_stage2_3_validation" in str(row.get("split", "")):
                rows.append(dict(row))
    return rows


def run_validate(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.legacy_metrics_jsonl:
        rows = _read_legacy_rows(args.legacy_metrics_jsonl)
        html_path = export_legacy_rows_html_report(rows, output_dir, title="IForward Validation v4 Legacy Rows")
        return [html_path]

    cfg = _load_cfg(args.config_file, args.opts)
    bundle = build_iforward_runtime_from_cfg(cfg, checkpoint=str(args.checkpoint or ""), device=args.device)
    plans = build_validation_v4_plans(
        cfg=cfg,
        dataset=bundle.dataset,
        max_entries=args.max_entries,
        frame_sets=_parse_csv(args.frame_sets),
        repair_permutations=args.repair_permutations,
        memory_ablation=_parse_csv(args.memory_ablation),
    )
    html_paths: list[str] = []
    for idx, plan in enumerate(plans):
        plan_dir = output_dir / f"{idx:04d}_{plan.plan_id}"
        scheduler = _make_scheduler(cfg, bundle.dataset)
        adapter = Stage3SchedulerAdapter(scheduler)
        runner = IForwardRunner(bundle.model, adapter, _convert_batch)
        recorder = TraceRecorder(plan_dir, record_images=True)
        trace = runner.run(
            plan,
            recorder,
            RunnerOptions.for_mode("validate", device=str(bundle.device), trigger_step=int(args.trigger_step)),
        )
        html_paths.append(export_html_report(trace, plan_dir, title=f"IForward Validation v4 {plan.episode.protocol_name}"))
    index = output_dir / "index.html"
    index.write_text(
        "<!doctype html><meta charset='utf-8'><h1>IForward Validation v4</h1><ul>"
        + "".join(
            f"<li><a href='{Path(path).parent.name}/index.html'>{Path(path).parent.name}</a></li>" for path in html_paths
        )
        + "</ul>",
        encoding="utf-8",
    )
    return [str(index), *html_paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IForward validation v4 validate-only reports.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scene_ids", default="", help="Reserved for future filtering; scheduler config currently controls scenes.")
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--frame_sets", default="seq10,seq24")
    parser.add_argument("--repair_permutations", type=int, default=None)
    parser.add_argument(
        "--memory_ablation",
        default="full,memory_off,memory_read_write,memory_freeze_write,memory_shuffle_state",
    )
    parser.add_argument("--trigger_step", type=int, default=0)
    parser.add_argument("--legacy_metrics_jsonl", default="")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()
    paths = run_validate(args)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
