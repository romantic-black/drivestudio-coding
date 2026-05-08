from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from datasets.multi_scene_dataset_v4 import BatchRequestV4

ImageRef = Tuple[int, int]


def make_refs_for_frames(*, frame_ids: List[int], camera_ids: List[int]) -> List[ImageRef]:
    return [(int(fid), int(cam)) for fid in frame_ids for cam in camera_ids]


def validate_source_refs(source_image_refs: List[ImageRef], camera_ids: List[int]) -> int:
    if len(source_image_refs) == 0:
        raise ValueError("source_image_refs must not be empty")
    source_frame_idx = int(source_image_refs[0][0])
    cam_set = set(int(x) for x in camera_ids)
    for fid, cam in source_image_refs:
        if int(fid) != int(source_frame_idx):
            raise ValueError(
                "all source refs must use the same source frame, got "
                f"source_frame_idx={source_frame_idx}, offending=({fid},{cam})"
            )
        if int(cam) not in cam_set:
            raise ValueError(f"source cam_id={cam} not in protocol camera_ids={sorted(cam_set)}")
    return int(source_frame_idx)


def validate_update_target_refs(
    *,
    update_target_image_refs: List[ImageRef],
    observed_frame_ids: List[int],
    camera_ids: List[int],
) -> None:
    if len(update_target_image_refs) == 0:
        raise ValueError("update_target_image_refs must not be empty")
    observed_set = set(int(x) for x in observed_frame_ids)
    cam_set = set(int(x) for x in camera_ids)
    for fid, cam in update_target_image_refs:
        if int(fid) not in observed_set:
            raise ValueError(
                "update target contains unobserved frame: "
                f"frame_idx={fid}, observed={sorted(observed_set)}"
            )
        if int(cam) not in cam_set:
            raise ValueError(
                f"update target cam_id={cam} not in protocol camera_ids={sorted(cam_set)}"
            )


def build_update_batch_from_refs(
    *,
    dataset: Any,
    scene_id: int,
    segment_id: int,
    source_image_refs: List[ImageRef],
    update_target_image_refs: List[ImageRef],
    observed_frame_ids: List[int],
    camera_ids: List[int],
    protocol_name: str,
    device: torch.device,
    enforce_target0_equals_source: bool = True,
) -> Dict[str, Any]:
    from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format

    source_frame_idx = validate_source_refs(source_image_refs, camera_ids)
    validate_update_target_refs(
        update_target_image_refs=update_target_image_refs,
        observed_frame_ids=observed_frame_ids,
        camera_ids=camera_ids,
    )

    raw = dataset.get_segment_batch_from_image_refs(
        BatchRequestV4(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            source_image_ref=(int(source_image_refs[0][0]), int(source_image_refs[0][1])),
            source_image_refs=[(int(f), int(c)) for f, c in source_image_refs],
            target_image_refs=[(int(f), int(c)) for f, c in update_target_image_refs],
            include_test=False,
        ),
        enforce_target0_equals_source=bool(enforce_target0_equals_source),
    )

    rm = dict(raw.get("request_meta") or {})
    rm.update(
        {
            "eval_protocol": str(protocol_name),
            "batch_role": "update",
            "source_image_refs": [(int(f), int(c)) for f, c in source_image_refs],
            "update_target_image_refs": [(int(f), int(c)) for f, c in update_target_image_refs],
            "observed_frame_ids": [int(x) for x in observed_frame_ids],
            "source_frame_idx": int(source_frame_idx),
        }
    )
    raw["request_meta"] = rm
    return convert_batch_to_minimal_format(
        raw,
        device=device,
        num_targets=len(update_target_image_refs),
        include_source_for_2d=True,
    )
