from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
import torch

from datasets.iforward_stage2_3.index_builder import build_stage2_3_index_from_dataset
from tools.train_minimal_streetforward_stage4_3_iforward_common import build_multi_scene_dataset_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IForward Stage2_3 optimizer-sequence raw-frame index")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("opts", nargs="*", help="Optional OmegaConf dotlist overrides")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    dataset = build_multi_scene_dataset_v4(cfg, device=torch.device("cpu"))
    index = build_stage2_3_index_from_dataset(dataset=dataset, cfg=cfg, output_dir=Path(args.output_dir))
    print(f"wrote Stage2_3 index to {args.output_dir}")
    print(f"fingerprint={index.fingerprint}")


if __name__ == "__main__":
    main()
