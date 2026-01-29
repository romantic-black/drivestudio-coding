"""
Record StreetForward Golden Baseline.

This script follows docs/trainers/StreetForward_Golden_Baseline_Design.md:
- real MultiSceneDataset batches (no stubs)
- sequential scheduler to make order deterministic
- fixed RNG seeds
- per-step summaries of loss / node states / offsets / grad norms
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch

from utils.streetforward_baseline import (
    BaselineStep,
    build_dataset,
    batch_plan_from_dataset,
    convert_batch_to_streetforward_format,
    default_baseline_meta,
    harvest_batch_cache,
    load_batch_cache,
    load_config,
    record_step,
    save_baseline,
    set_deterministic_seed,
)


def _default_output(config_path: str, seed: int, steps: int, device: torch.device) -> Path:
    cfg_name = Path(config_path).stem
    return Path("docs/trainers/golden") / f"streetforward_golden_{cfg_name}_seed{seed}_steps{steps}_{device.type}.json"


def run_recording(
    config_path: str,
    steps: int,
    seed: int,
    device: torch.device,
    batches_per_segment: int,
    segment_order: str,
    scene_order: str,
    shuffle_segments: bool,
    preload_next_scene: bool,
    output: Path,
    batch_cache_path: Optional[Path] = None,
    harvest_if_missing: bool = False,
    plan_scenes: int = 2,
    plan_segments: int = 2,
    plan_batches: int = 2,
) -> Path:
    # Import StreetForwardTrainer without loading trainers/__init__.py (and thus base, pytorch3d, etc.)
    def _load_streetforward_trainer():
        import importlib.util
        import sys
        from types import ModuleType
        root = Path(__file__).resolve().parent.parent
        path = root / "models" / "trainers" / "streetforward.py"
        spec = importlib.util.spec_from_file_location("models.trainers.streetforward", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        if "models.trainers" not in sys.modules:
            sys.modules["models.trainers"] = ModuleType("models.trainers")
        mod.__package__ = "models.trainers"
        sys.modules["models.trainers.streetforward"] = mod
        spec.loader.exec_module(mod)
        return mod.StreetForwardTrainer

    StreetForwardTrainer = _load_streetforward_trainer()
    set_deterministic_seed(seed)
    cfg = load_config(config_path)
    cfg.config_path = os.path.abspath(config_path)

    batches_from_cache: Optional[List[Dict]] = None
    cache_meta: Optional[Dict] = None
    if batch_cache_path is not None:
        if batch_cache_path.exists():
            cache_meta, batches_from_cache = load_batch_cache(batch_cache_path)
        elif harvest_if_missing:
            dataset_for_plan = build_dataset(cfg, device)
            plan = batch_plan_from_dataset(
                dataset_for_plan,
                max_scenes=plan_scenes,
                segments_per_scene=plan_segments,
                batches_per_segment=plan_batches,
            )
            cache_meta = harvest_batch_cache(
                cfg,
                device,
                seed,
                plan,
                batch_cache_path,
                include_test=False,
            )
            _, batches_from_cache = load_batch_cache(batch_cache_path)

    dataset = None
    scheduler = None
    if batches_from_cache is None:
        dataset = build_dataset(cfg, device)
        scheduler = dataset.create_scheduler(
            batches_per_segment=batches_per_segment,
            segment_order=segment_order,
            scene_order=scene_order,
            shuffle_segments=shuffle_segments,
            preload_next_scene=preload_next_scene,
            include_test=False,
        )
    batch_iter = batches_from_cache if batches_from_cache is not None else None
    trainer = StreetForwardTrainer(config=cfg, device=device)
    trainer.train()

    per_step: List[BaselineStep] = []
    scene_segment_sequence: List[Tuple[int, int]] = []

    try:
        for step_idx in range(steps):
            if batch_iter is not None:
                if step_idx >= len(batch_iter):
                    break
                batch = batch_iter[step_idx]
            else:
                try:
                    batch = scheduler.next_batch()
                except StopIteration:
                    break
            scene_segment_sequence.append(
                (
                    int(batch["scene_id"]) if not isinstance(batch["scene_id"], torch.Tensor) else int(batch["scene_id"].item()),
                    int(batch["segment_id"]) if not isinstance(batch["segment_id"], torch.Tensor) else int(batch["segment_id"].item()),
                )
            )
            street_batch = convert_batch_to_streetforward_format(batch, device)
            result = trainer.train_iter(
                batch=street_batch,
                apply_update=True,
                update_state=True,
            )
            per_step.append(record_step(trainer, street_batch, result, step_idx))
    finally:
        if scheduler is not None:
            scheduler.shutdown()

    baseline = {
        "meta": default_baseline_meta(
            cfg,
            seed,
            len(per_step),
            device,
            scene_segment_sequence,
            scheduler_kwargs={
                "batches_per_segment": batches_per_segment,
                "segment_order": segment_order,
                "scene_order": scene_order,
                "shuffle_segments": shuffle_segments,
                "preload_next_scene": preload_next_scene,
            },
            batch_cache_path=str(batch_cache_path) if batch_cache_path is not None else None,
            batch_plan=cache_meta.get("plan") if cache_meta else None,
        ),
        "per_step": [s.to_dict() for s in per_step],
    }
    save_baseline(baseline, output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Record StreetForward Golden Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/streetforward/multi_scene.yaml",
        help="Path to StreetForward training config",
    )
    parser.add_argument("--steps", type=int, default=8, help="Number of train_iter steps to record (>=8 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. cuda or cpu (default: auto)")
    parser.add_argument("--batches-per-segment", type=int, default=2, help="Batches per segment for scheduler")
    parser.add_argument(
        "--segment-order",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
        help="Segment traversal order",
    )
    parser.add_argument(
        "--scene-order",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
        help="Scene traversal order",
    )
    parser.add_argument(
        "--shuffle-segments",
        action="store_true",
        help="Shuffle segments inside a scene (default: False)",
    )
    parser.add_argument(
        "--preload-next-scene",
        action="store_true",
        help="Enable scheduler background preload of next scene",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output baseline file path (default: docs/trainers/golden/...)",
    )
    parser.add_argument(
        "--batch-cache",
        type=str,
        default=None,
        help="Optional path to batch cache (.pt). If provided and exists, recording reads batches from cache.",
    )
    parser.add_argument(
        "--harvest-batch-cache",
        action="store_true",
        help="If set and --batch-cache provided (or default path), harvest batches first when cache is missing.",
    )
    parser.add_argument("--plan-scenes", type=int, default=2, help="Plan: number of scenes when harvesting cache")
    parser.add_argument("--plan-segments", type=int, default=2, help="Plan: segments per scene when harvesting cache")
    parser.add_argument("--plan-batches", type=int, default=2, help="Plan: batches per segment when harvesting cache")

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output) if args.output else _default_output(args.config, args.seed, args.steps, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_cache_path = Path(args.batch_cache) if args.batch_cache else None

    recorded = run_recording(
        config_path=args.config,
        steps=args.steps,
        seed=args.seed,
        device=device,
        batches_per_segment=args.batches_per_segment,
        segment_order=args.segment_order,
        scene_order=args.scene_order,
        shuffle_segments=bool(args.shuffle_segments),
        preload_next_scene=bool(args.preload_next_scene),
        output=output_path,
        batch_cache_path=batch_cache_path,
        harvest_if_missing=bool(args.harvest_batch_cache),
        plan_scenes=args.plan_scenes,
        plan_segments=args.plan_segments,
        plan_batches=args.plan_batches,
    )
    print(f"[baseline] saved to {recorded}")


if __name__ == "__main__":
    main()
