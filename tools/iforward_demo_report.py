from __future__ import annotations

import argparse
from pathlib import Path

from models.iforward.demo.report_builder import build_demo_report
from models.iforward.protocols.demo_recipes import build_demo_v0_plans, iforward_demo_cfg, make_demo_scheduler
from tools.iforward_validate_v4 import _convert_batch, _load_cfg, _parse_csv, build_iforward_runtime_from_cfg


def run_demo_report(args: argparse.Namespace) -> str:
    cfg = _load_cfg(args.config_file, args.opts)
    demo_cfg = iforward_demo_cfg(cfg)
    recipe = str(args.recipe or demo_cfg["default_recipe"])
    output_dir = str(args.output_dir or demo_cfg["output_dir"] or f"iforward_demo_{recipe}")
    memory_modes = _parse_csv(args.memory_ablation) or list(demo_cfg["memory_ablation"])
    seed = int(args.seed if args.seed is not None else demo_cfg["seed"])
    bundle = build_iforward_runtime_from_cfg(cfg, checkpoint=str(args.checkpoint or ""), device=args.device)
    recipe_result = build_demo_v0_plans(
        cfg=cfg,
        dataset=bundle.dataset,
        recipe=recipe,
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(seed),
        memory_ablation=memory_modes,
    )
    scheduler = make_demo_scheduler(
        cfg=cfg,
        dataset=bundle.dataset,
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(seed),
    )
    result = build_demo_report(
        recipe=recipe_result.recipe,
        plans=recipe_result.plans,
        model=bundle.model,
        scheduler=scheduler,
        output_dir=Path(output_dir),
        device=bundle.device,
        trigger_step=int(args.trigger_step),
        convert_batch_to_minimal_format=_convert_batch,
    )
    return result.index_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static IForward Demo v0 HTML report.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--scene_id", type=int, required=True)
    parser.add_argument("--segment_id", type=int, required=True)
    parser.add_argument("--recipe", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trigger_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--memory_ablation",
        default="",
        help="Comma-separated modes for memory_ablation_showcase. Defaults to iforward_demo.memory_ablation.",
    )
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()
    print(run_demo_report(args))


if __name__ == "__main__":
    main()
