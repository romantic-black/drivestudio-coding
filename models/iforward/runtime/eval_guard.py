from __future__ import annotations

from typing import Any, Mapping


FROZEN_FEEDBACK_EVAL_MODE = "frozen_no_grad"
FROZEN_FEEDBACK_RUNNER_MODES = frozenset({"validate", "demo", "replay"})


def apply_frozen_feedback_eval_metadata(batch: dict[str, Any]) -> dict[str, Any]:
    """Mark an evaluation batch as graph-free without changing its train metadata.

    Both metadata locations are part of the Stage 3 scheduler compatibility ABI.
    Some conversion paths materialize only the top-level location while older
    schedulers read the nested one, so evaluation runners must populate both.
    """

    request_meta = dict(batch.get("request_meta", {}) or {})
    request_meta["observation_feedback_eval_mode"] = FROZEN_FEEDBACK_EVAL_MODE
    batch["request_meta"] = request_meta

    iforward_meta = batch.get("_iforward", None)
    if isinstance(iforward_meta, dict):
        nested_request_meta = dict(iforward_meta.get("request_meta", {}) or {})
        nested_request_meta["observation_feedback_eval_mode"] = FROZEN_FEEDBACK_EVAL_MODE
        iforward_meta["request_meta"] = nested_request_meta
    return batch


def parameter_version_snapshot(model: Any) -> dict[str, int]:
    """Capture PyTorch's in-place mutation counter for every model parameter."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return {}
    return {
        str(name): int(getattr(param, "_version", 0))
        for name, param in named_parameters()
    }


def changed_parameter_versions(
    model: Any,
    before: Mapping[str, int],
) -> dict[str, tuple[int | None, int | None]]:
    after = parameter_version_snapshot(model)
    names = sorted(set(str(name) for name in before) | set(after))
    return {
        name: (before.get(name), after.get(name))
        for name in names
        if before.get(name) != after.get(name)
    }


def assert_parameter_versions_unchanged(model: Any, before: Mapping[str, int]) -> None:
    changed = changed_parameter_versions(model, before)
    if changed:
        preview = ", ".join(
            f"{name}:{old}->{new}"
            for name, (old, new) in list(changed.items())[:20]
        )
        raise RuntimeError(
            "IForward evaluation mutated model parameters; validation/demo/replay "
            f"must be read-only ({preview})"
        )


__all__ = [
    "FROZEN_FEEDBACK_EVAL_MODE",
    "FROZEN_FEEDBACK_RUNNER_MODES",
    "apply_frozen_feedback_eval_metadata",
    "assert_parameter_versions_unchanged",
    "changed_parameter_versions",
    "parameter_version_snapshot",
]
