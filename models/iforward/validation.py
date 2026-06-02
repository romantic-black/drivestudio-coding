from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_IFORWARD_ABLATIONS = (
    "full",
    "zero_all",
    "zero_point",
    "zero_cell",
    "zero_global",
    "drop_short_window",
    "freeze_write",
    "shuffle_memory",
)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: List[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _clone_state_for_ablation(state: Any) -> Any:
    if state is None:
        return None
    if hasattr(state, "detach_for_next_rollout"):
        return state.detach_for_next_rollout()
    return copy.deepcopy(state)


def validate_iforward_memory_ablation(
    *,
    model: Any,
    rollout_batches: Sequence[Dict[str, Any]],
    initial_state: Optional[Any] = None,
    ablations: Iterable[str] = DEFAULT_IFORWARD_ABLATIONS,
) -> List[Dict[str, Any]]:
    """Run memory ablations from the same initial carried state.

    This helper intentionally does not call `train_step`; it evaluates the
    exact `forward_rollout(..., ablation=mode)` path and carries each mode's
    own detached state through the provided rollout sequence.
    """

    batches = list(rollout_batches)
    if not batches:
        raise ValueError("validate_iforward_memory_ablation requires non-empty rollout_batches.")

    rows: List[Dict[str, Any]] = []
    for mode in [str(x) for x in list(ablations)]:
        carried = _clone_state_for_ablation(initial_state)
        current_psnr: List[float] = []
        history_rollout_psnr: List[float] = []
        short_window_psnr: List[float] = []
        nearby_psnr: List[float] = []
        losses: List[float] = []
        final_stats: Dict[str, Any] = {}

        for batch in batches:
            out = model.forward_rollout(batch, carried_state=carried, ablation=mode)
            losses.append(_finite_float(out.loss.detach().item()))
            stats = dict(out.stats or {})
            final_stats = stats
            current_psnr.append(_finite_float(stats.get("current_psnr")))
            history_rollout_psnr.append(_finite_float(stats.get("history_rollout_psnr")))
            short_window_psnr.append(_finite_float(stats.get("short_window_history_psnr")))
            nearby_psnr.append(_finite_float(stats.get("nearby_psnr")))
            if bool(out.resolved.carry_scene_state_after_rollout) and not bool(out.resolved.episode_end_after_rollout):
                carried = out.next_state.detach_for_next_rollout()
            else:
                carried = None

        cur = _mean(current_psnr)
        hist = _mean(history_rollout_psnr)
        short = _mean(short_window_psnr)
        near = _mean(nearby_psnr)
        rows.append(
            {
                "mode": mode,
                "num_rollouts": int(len(batches)),
                "loss": _mean(losses),
                "current_psnr": cur,
                "history_rollout_psnr": hist,
                "history_short_window_psnr": short,
                "nearby_psnr": near,
                "retention_gap_rollout": cur - hist if math.isfinite(cur) and math.isfinite(hist) else float("nan"),
                "retention_gap_short_window": cur - short if math.isfinite(cur) and math.isfinite(short) else float("nan"),
                "final_current_psnr": _finite_float(final_stats.get("current_psnr")),
                "final_history_rollout_psnr": _finite_float(final_stats.get("history_rollout_psnr")),
                "final_history_short_window_psnr": _finite_float(final_stats.get("short_window_history_psnr")),
                "final_nearby_psnr": _finite_float(final_stats.get("nearby_psnr")),
            }
        )
    return rows


__all__ = ["DEFAULT_IFORWARD_ABLATIONS", "validate_iforward_memory_ablation"]
