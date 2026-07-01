from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.plan import EpisodePlan
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import TraceRecorder
from models.iforward.validation_v4.html_exporter import export_html_report
from tools.iforward_validate_v4 import _convert_batch, _load_cfg, _make_scheduler, build_iforward_runtime_from_cfg


def run_replay(args: argparse.Namespace) -> str:
    cfg = _load_cfg(args.config_file, args.opts)
    bundle = build_iforward_runtime_from_cfg(cfg, checkpoint=str(args.checkpoint or ""), device=args.device)
    with open(args.plan_json, "r", encoding="utf-8") as fh:
        plan = EpisodePlan.from_json_dict(json.load(fh))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler = _make_scheduler(cfg, bundle.dataset)
    adapter = Stage3SchedulerAdapter(scheduler)
    runner = IForwardRunner(bundle.model, adapter, _convert_batch)
    recorder = TraceRecorder(output_dir, record_images=True)
    trace = runner.run(
        plan,
        recorder,
        RunnerOptions.for_mode("replay", device=str(bundle.device), trigger_step=int(args.trigger_step)),
    )
    return export_html_report(trace, output_dir, title=f"IForward Plan Replay {plan.plan_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an IForward runtime EpisodePlan JSON.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trigger_step", type=int, default=0)
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()
    print(run_replay(args))


if __name__ == "__main__":
    main()
