from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/root/autodl-tmp/streetforward_assets_pt60w_40w")
DEFAULT_BATCH_ROOT = Path("/root/autodl-tmp/outputs/stage5_6_demo_video_batch_gt30")
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs/viewer/demo_minimal_streetforward_stage5_6_video.yaml"
DEFAULT_VIDEO_SCRIPT = REPO_ROOT / "tools/demo_minimal_streetforward_stage5_6_video.py"


@dataclass(frozen=True)
class SegmentJob:
    scene_id: int
    segment_id: int
    num_frames: int
    segment_asset_id: str
    scene_asset_id: str


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_jobs_from_assets(*, asset_root: Path, min_frames: int, frame_key: str) -> List[SegmentJob]:
    segment_root = Path(asset_root) / "segment_pool"
    if not segment_root.is_dir():
        raise FileNotFoundError(f"segment_pool not found under asset_root={asset_root}")

    jobs: List[SegmentJob] = []
    for seg_dir in sorted(segment_root.glob("seg-*")):
        manifest_path = seg_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        stats = manifest.get("stats") or {}
        if frame_key not in stats:
            continue
        num_frames = int(stats[frame_key])
        if num_frames <= int(min_frames):
            continue
        jobs.append(
            SegmentJob(
                scene_id=int(manifest["scene_id"]),
                segment_id=int(manifest["segment_id"]),
                num_frames=int(num_frames),
                segment_asset_id=str(manifest.get("asset_id") or seg_dir.name),
                scene_asset_id=str(manifest.get("parent_scene_asset_id") or ""),
            )
        )
    return jobs


def _load_jobs_from_tsv(*, path: Path, min_frames: int) -> List[SegmentJob]:
    jobs: List[SegmentJob] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            num_frames = int(row["num_frames"])
            if num_frames <= int(min_frames):
                continue
            jobs.append(
                SegmentJob(
                    scene_id=int(row["scene_id"]),
                    segment_id=int(row["segment_id"]),
                    num_frames=int(num_frames),
                    segment_asset_id=str(row.get("segment_asset_id") or ""),
                    scene_asset_id=str(row.get("scene_asset_id") or ""),
                )
            )
    return jobs


def _parse_scene_filter(raw: str) -> Optional[Set[int]]:
    text = str(raw or "").strip()
    if not text:
        return None
    out: Set[int] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    return out


def _sort_jobs(jobs: Sequence[SegmentJob], order: str) -> List[SegmentJob]:
    mode = str(order).strip().lower()
    if mode == "frames_desc":
        return sorted(jobs, key=lambda j: (-int(j.num_frames), int(j.scene_id), int(j.segment_id)))
    if mode == "frames_asc":
        return sorted(jobs, key=lambda j: (int(j.num_frames), int(j.scene_id), int(j.segment_id)))
    if mode != "scene":
        raise ValueError("--order must be one of: scene, frames_desc, frames_asc")
    return sorted(jobs, key=lambda j: (int(j.scene_id), int(j.segment_id)))


def _job_tag(job: SegmentJob) -> str:
    return f"scene{int(job.scene_id):06d}_seg{int(job.segment_id):06d}_frames{int(job.num_frames):03d}"


def _metadata_path(batch_root: Path, job: SegmentJob) -> Path:
    tag = _job_tag(job)
    return Path(batch_root) / "videos" / tag / f"{tag}_metadata.json"


def _write_job_list(path: Path, jobs: Sequence[SegmentJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "segment_id", "num_frames", "segment_asset_id", "scene_asset_id"],
            delimiter="\t",
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "scene_id": f"{int(job.scene_id):06d}",
                    "segment_id": int(job.segment_id),
                    "num_frames": int(job.num_frames),
                    "segment_asset_id": str(job.segment_asset_id),
                    "scene_asset_id": str(job.scene_asset_id),
                }
            )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _extra_opts(raw: Sequence[str]) -> List[str]:
    opts = list(raw or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    return opts


def _build_command(args: argparse.Namespace, job: SegmentJob, extra_opts: Sequence[str]) -> List[str]:
    tag = _job_tag(job)
    batch_root = Path(args.batch_root)
    run_dir = batch_root / "runs" / tag
    video_dir = batch_root / "videos" / tag
    cmd = [
        sys.executable,
        str(Path(args.video_script)),
        "--config_file",
        str(Path(args.config_file)),
        "--scene_id",
        str(int(job.scene_id)),
        "--segment_id",
        str(int(job.segment_id)),
        "--sequence_start_pos",
        str(int(args.sequence_start_pos)),
    ]
    if str(args.camera_mode).strip():
        cmd += ["--camera_mode", str(args.camera_mode).strip()]
    if str(args.device).strip():
        cmd += ["--device", str(args.device).strip()]
    if str(args.ckpt).strip():
        cmd += ["--ckpt", str(args.ckpt).strip()]
    if str(args.ckpt_load_mode).strip():
        cmd += ["--ckpt_load_mode", str(args.ckpt_load_mode).strip()]

    overrides = [
        f"output_name={tag}",
        f"logging.log_dir={run_dir}",
        f"video.output.dir={video_dir}",
        f"video.output.name={tag}",
        f"batch_eval.dataset.start_at.scene_id={int(job.scene_id)}",
        f"batch_eval.dataset.start_at.segment_id={int(job.segment_id)}",
        "batch_eval.dataset.start_at.frame_id=0",
    ]
    if args.max_windows is not None:
        overrides.append(f"video.reconstruction.max_windows={int(args.max_windows)}")
    if not bool(args.save_all_images):
        overrides += [
            "video.output.save_all_images=false",
            "video.output.save_png_frames=false",
        ]
    overrides += list(extra_opts)
    return cmd + overrides


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run Stage5_6 demo video export for every segment whose manifest "
            "frame count is greater than the threshold."
        )
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--segment-list", type=Path, default=None, help="Optional TSV produced by a previous scan.")
    parser.add_argument("--min-frames", type=int, default=30, help="Keep segments with frame_key > min_frames.")
    parser.add_argument("--frame-key", type=str, default="num_train_frames")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--video-script", type=Path, default=DEFAULT_VIDEO_SCRIPT)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--order", type=str, default="scene", choices=["scene", "frames_desc", "frames_asc"])
    parser.add_argument("--only-scenes", type=str, default="", help="Comma list/ranges, e.g. 0,24,99-103.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--sequence-start-pos", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None, help="Forwarded to video.reconstruction.max_windows.")
    parser.add_argument("--camera-mode", type=str, default="", help="Forwarded to --camera_mode.")
    parser.add_argument("--device", type=str, default="", help="Forwarded to --device, e.g. cuda:0.")
    parser.add_argument("--ckpt", type=str, default="", help="Forwarded to --ckpt.")
    parser.add_argument("--ckpt-load-mode", type=str, default="", help="Forwarded to --ckpt_load_mode.")
    parser.add_argument("--save-all-images", action="store_true", help="Keep frames_all images; default only writes videos/metadata.")
    parser.add_argument("--rerun-completed", action="store_true", help="Do not skip jobs with existing metadata.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "video_opts",
        nargs=argparse.REMAINDER,
        help="Extra OmegaConf overrides for the video script. Use -- before these if needed.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    batch_root = Path(args.batch_root)
    extra_opts = _extra_opts(args.video_opts)

    if args.segment_list is not None:
        jobs = _load_jobs_from_tsv(path=Path(args.segment_list), min_frames=int(args.min_frames))
    else:
        jobs = _load_jobs_from_assets(
            asset_root=Path(args.asset_root),
            min_frames=int(args.min_frames),
            frame_key=str(args.frame_key),
        )

    scene_filter = _parse_scene_filter(args.only_scenes)
    if scene_filter is not None:
        jobs = [job for job in jobs if int(job.scene_id) in scene_filter]
    jobs = _sort_jobs(jobs, str(args.order))
    if int(args.offset) > 0:
        jobs = jobs[int(args.offset) :]
    if int(args.limit) > 0:
        jobs = jobs[: int(args.limit)]

    batch_root.mkdir(parents=True, exist_ok=True)
    selected_path = batch_root / "selected_segments.tsv"
    status_path = batch_root / "status.jsonl"
    _write_job_list(selected_path, jobs)

    print(f"selected_jobs={len(jobs)}")
    print(f"selected_segments={selected_path}")
    print(f"status_jsonl={status_path}")
    if len(jobs) == 0:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for idx, job in enumerate(jobs, start=1):
        tag = _job_tag(job)
        metadata_path = _metadata_path(batch_root, job)
        if metadata_path.exists() and not bool(args.rerun_completed):
            skipped += 1
            print(f"[{idx}/{len(jobs)}] skip completed {tag}")
            _append_jsonl(
                status_path,
                {
                    "status": "skipped",
                    "tag": tag,
                    "scene_id": int(job.scene_id),
                    "segment_id": int(job.segment_id),
                    "num_frames": int(job.num_frames),
                    "metadata_path": str(metadata_path),
                    "time_unix": time.time(),
                },
            )
            continue

        cmd = _build_command(args, job, extra_opts)
        print(f"[{idx}/{len(jobs)}] run {tag}")
        print(shlex.join(cmd))
        if bool(args.dry_run):
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        elapsed = time.time() - t0
        record = {
            "status": "ok" if int(result.returncode) == 0 else "failed",
            "returncode": int(result.returncode),
            "tag": tag,
            "scene_id": int(job.scene_id),
            "segment_id": int(job.segment_id),
            "num_frames": int(job.num_frames),
            "elapsed_seconds": float(elapsed),
            "metadata_path": str(metadata_path),
            "command": cmd,
            "time_unix": time.time(),
        }
        _append_jsonl(status_path, record)
        if int(result.returncode) == 0:
            completed += 1
        else:
            failed += 1
            if bool(args.stop_on_failure):
                break

    elapsed_total = time.time() - started_at
    print(
        "summary "
        f"completed={completed} skipped={skipped} failed={failed} "
        f"dry_run={bool(args.dry_run)} elapsed_seconds={elapsed_total:.1f}"
    )


if __name__ == "__main__":
    main()
