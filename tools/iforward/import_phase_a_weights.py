from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch


def _state_dict_from_payload(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if all(torch.is_tensor(v) for v in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("checkpoint payload does not contain a model state_dict")


def _candidate_keys_for_iforward(key: str) -> Tuple[List[str], bool]:
    if key.startswith("model.phase_a_runtime."):
        suffix = key[len("model.phase_a_runtime.") :]
        return [key, f"phase_a_runtime.{suffix}"], True
    if key.startswith("phase_a_runtime."):
        suffix = key[len("phase_a_runtime.") :]
        return [f"model.{key}", key], True
    if key.startswith("runtime."):
        suffix = key[len("runtime.") :]
        return [f"model.phase_a_runtime.{suffix}", f"phase_a_runtime.{suffix}"], True
    prefixes = (
        "stage6_struct_event_decoder.",
        "stage6_posterior_updater.",
        "image_feature_extractor.",
        "backprojector",
        "node_state",
    )
    if key.startswith(prefixes):
        return [f"model.phase_a_runtime.{key}", f"phase_a_runtime.{key}"], True
    return [f"model.phase_a_runtime.{key}", f"phase_a_runtime.{key}"], False


def import_phase_a_weights(
    *,
    source_checkpoint: str,
    target_checkpoint: str,
    output_checkpoint: str,
    strict_target_load: bool = False,
) -> Dict[str, Any]:
    source_payload = torch.load(source_checkpoint, map_location="cpu")
    target_payload = torch.load(target_checkpoint, map_location="cpu")
    if not isinstance(source_payload, dict) or not isinstance(target_payload, dict):
        raise ValueError("source and target checkpoints must be dict payloads")
    source_sd = _state_dict_from_payload(source_payload)
    target_sd = OrderedDict(_state_dict_from_payload(target_payload))

    imported = []
    skipped_shape = []
    skipped_missing = []
    for key, value in source_sd.items():
        candidates, likely = _candidate_keys_for_iforward(str(key))
        mapped = next((candidate for candidate in candidates if candidate in target_sd), candidates[0])
        if mapped not in target_sd:
            if likely:
                skipped_missing.append((str(key), candidates))
            continue
        if tuple(target_sd[mapped].shape) != tuple(value.shape):
            skipped_shape.append((str(key), mapped, tuple(value.shape), tuple(target_sd[mapped].shape)))
            continue
        target_sd[mapped] = value.detach().cpu().clone()
        imported.append((str(key), mapped))

    if strict_target_load and (skipped_shape or not imported):
        raise RuntimeError(
            f"phase-A import failed strict checks: imported={len(imported)} shape_mismatch={len(skipped_shape)}"
        )
    out_payload = dict(target_payload)
    if "model_state_dict" in out_payload:
        out_payload["model_state_dict"] = target_sd
    elif "state_dict" in out_payload:
        out_payload["state_dict"] = target_sd
    else:
        out_payload["model_state_dict"] = target_sd
    report = {
        "format": "iforward_phase_a_import_v1",
        "source_checkpoint": source_checkpoint,
        "target_checkpoint": target_checkpoint,
        "num_imported": len(imported),
        "num_skipped_missing": len(skipped_missing),
        "num_skipped_shape": len(skipped_shape),
        "imported": imported[:200],
        "skipped_missing": skipped_missing[:200],
        "skipped_shape": skipped_shape[:200],
    }
    out_payload["iforward_phase_a_import"] = report
    torch.save(out_payload, output_checkpoint)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Stage6 Phase A weights into an IForward checkpoint.")
    parser.add_argument("--source_phase_a", required=True, help="Stage6 Phase A checkpoint path.")
    parser.add_argument("--target_iforward", required=True, help="Fresh or existing IForward checkpoint path.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--strict_target_load", action="store_true")
    args = parser.parse_args()
    report = import_phase_a_weights(
        source_checkpoint=args.source_phase_a,
        target_checkpoint=args.target_iforward,
        output_checkpoint=args.output,
        strict_target_load=bool(args.strict_target_load),
    )
    print(
        "Imported {num_imported} tensors; skipped_missing={num_skipped_missing}; "
        "skipped_shape={num_skipped_shape}".format(**report)
    )


if __name__ == "__main__":
    main()
