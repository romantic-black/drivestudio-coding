from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


DEFAULT_GRADIENT_GROUPS: dict[str, tuple[str, ...]] = {
    "posterior_updater": ("posterior_updater", "stage6_posterior_updater"),
    "child_decoder": ("child_decoder", "biggs_child_decoder", "struct_event_decoder"),
    "parent_ptv3": ("parent_spatial", "parent_ptv3"),
    "gdkv_adapter": ("parent_temporal_adapter", "adapter", "token_builder"),
    "residual_2d": ("residual_unet", "measurement_frontend"),
}


def gradient_conflict_cosines(
    *,
    model: torch.nn.Module,
    losses: Mapping[str, torch.Tensor],
    loss_keys: Sequence[str] = ("current", "in_rollout_history", "history_damage"),
    groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Measure loss-gradient cosine by module-name groups for one live graph.

    The caller must pass losses from a forward pass whose graph is still alive.
    This helper does not call backward and does not mutate gradients on params.
    """

    selected_losses = {
        str(key): value
        for key, value in dict(losses).items()
        if str(key) in {str(x) for x in loss_keys} and torch.is_tensor(value) and bool(value.requires_grad)
    }
    if len(selected_losses) < 2:
        return {}
    group_map = {str(k): tuple(str(x) for x in v) for k, v in dict(groups or DEFAULT_GRADIENT_GROUPS).items()}
    named_params = [(name, param) for name, param in model.named_parameters() if bool(param.requires_grad)]
    if not named_params:
        return {}
    params = [param for _, param in named_params]
    grads_by_loss: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for key, loss in selected_losses.items():
        grads_by_loss[key] = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

    out: dict[str, float] = {}
    keys = list(selected_losses.keys())
    for i, lhs in enumerate(keys):
        for rhs in keys[i + 1 :]:
            out[f"all/{lhs}_vs_{rhs}"] = _cosine(
                _flatten_grads(grads_by_loss[lhs]),
                _flatten_grads(grads_by_loss[rhs]),
            )
            for group_name, tokens in group_map.items():
                indices = [idx for idx, (name, _) in enumerate(named_params) if any(token in name for token in tokens)]
                if not indices:
                    continue
                out[f"{group_name}/{lhs}_vs_{rhs}"] = _cosine(
                    _flatten_grads(grads_by_loss[lhs], indices=indices),
                    _flatten_grads(grads_by_loss[rhs], indices=indices),
                )
    return out


def _flatten_grads(grads: Sequence[torch.Tensor | None], *, indices: Sequence[int] | None = None) -> torch.Tensor:
    selected = list(indices) if indices is not None else list(range(len(grads)))
    parts = []
    for idx in selected:
        grad = grads[int(idx)]
        if grad is not None:
            parts.append(grad.detach().float().reshape(-1))
    if not parts:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if int(lhs.numel()) == 0 or int(rhs.numel()) == 0:
        return 0.0
    n = min(int(lhs.numel()), int(rhs.numel()))
    lhs = lhs[:n]
    rhs = rhs[:n]
    denom = torch.linalg.vector_norm(lhs) * torch.linalg.vector_norm(rhs)
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return float(torch.dot(lhs, rhs).div(denom).detach().cpu().item())


__all__ = ["DEFAULT_GRADIENT_GROUPS", "gradient_conflict_cosines"]
