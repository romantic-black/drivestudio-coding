from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


_MISSING = object()


def _cfg_get(node: Any, key: str, default: Any = _MISSING) -> Any:
    if node is None:
        if default is _MISSING:
            raise KeyError(f"missing required config key: {key}")
        return default
    if isinstance(node, dict):
        if key in node:
            return node[key]
        if default is _MISSING:
            raise KeyError(f"missing required config key: {key}")
        return default
    if hasattr(node, "get"):
        out = node.get(key, _MISSING)
        if out is not _MISSING:
            return out
    if hasattr(node, key):
        return getattr(node, key)
    if default is _MISSING:
        raise KeyError(f"missing required config key: {key}")
    return default


def _as_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    if hasattr(node, "keys"):
        return {str(k): node[k] for k in node.keys()}
    raise TypeError(f"config node is not mapping-like: {type(node)}")


def warmup_cosine_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    warmup_start_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        t = float(step) / float(max(1, warmup_steps))
        return float(warmup_start_ratio) + t * (1.0 - float(warmup_start_ratio))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine


class StreetForwardWarmupCosineLR:
    def __init__(self, optimizer: torch.optim.Optimizer, cfg: Any, *, start_step: int = 0):
        self.optimizer = optimizer
        self.cfg = cfg
        self.global_step = int(start_step)
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
        self.set_step(self.global_step)

    def factor(self, step: int) -> float:
        return warmup_cosine_factor(
            int(step),
            warmup_steps=int(_cfg_get(self.cfg, "warmup_steps")),
            total_steps=int(_cfg_get(self.cfg, "total_steps")),
            min_lr_ratio=float(_cfg_get(self.cfg, "min_lr_ratio")),
            warmup_start_ratio=float(_cfg_get(self.cfg, "warmup_start_ratio")),
        )

    def set_step(self, step: int) -> None:
        self.global_step = int(step)
        fac = float(self.factor(self.global_step))
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * fac

    def step(self) -> None:
        self.set_step(self.global_step + 1)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "cfg": _as_dict(self.cfg),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.set_step(int(state.get("global_step", 0)))


@dataclass
class _NoDecayCfg:
    enable: bool
    name_keywords: Tuple[str, ...]
    ndim_leq: int


def _parse_no_decay_cfg(optimizer_cfg: Any) -> _NoDecayCfg:
    nd_cfg = _cfg_get(optimizer_cfg, "no_weight_decay", {})
    if not _cfg_get(nd_cfg, "enable", True):
        return _NoDecayCfg(enable=False, name_keywords=(), ndim_leq=-1)
    raw_keywords = _cfg_get(nd_cfg, "name_keywords", [])
    name_keywords = tuple(str(x) for x in list(raw_keywords))
    ndim_leq = int(_cfg_get(nd_cfg, "ndim_leq", 1))
    return _NoDecayCfg(enable=True, name_keywords=name_keywords, ndim_leq=ndim_leq)


def _use_no_weight_decay(name: str, p: torch.nn.Parameter, cfg: _NoDecayCfg) -> bool:
    if not cfg.enable:
        return False
    if int(p.ndim) <= int(cfg.ndim_leq):
        return True
    return any(kw in name for kw in cfg.name_keywords)


def _prefix_match(name: str, prefix: str) -> bool:
    p = str(prefix).strip()
    if not p:
        return False
    if name == p:
        return True
    if p.endswith("."):
        return name.startswith(p)
    return name.startswith(p + ".")


def _match_group(name: str, match_cfg: Any) -> bool:
    if match_cfg is None:
        return False
    prefixes = list(_cfg_get(match_cfg, "prefixes", []))
    contains = list(_cfg_get(match_cfg, "contains", []))
    for p in prefixes:
        if _prefix_match(name, str(p)):
            return True
    for c in contains:
        if str(c) in name:
            return True
    return False


def _group_order(groups_cfg: Dict[str, Any]) -> List[str]:
    out = [k for k in groups_cfg.keys() if str(k) != "default"]
    if "default" in groups_cfg:
        out.append("default")
    return out


def _build_meta_signature(param_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_groups": len(param_groups),
        "groups": [
            {
                "name": str(g.get("name", "")),
                "num_params": int(len(g.get("param_names", []))),
                "param_names": list(g.get("param_names", [])),
            }
            for g in param_groups
        ],
    }


def optimizer_group_signature(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    out = {"num_groups": len(optimizer.param_groups), "groups": []}
    for g in optimizer.param_groups:
        pnames = list(g.get("param_names", []))
        out["groups"].append(
            {
                "name": str(g.get("name", "")),
                "num_params": int(len(pnames)),
                "param_names": pnames,
            }
        )
    return out


def build_streetforward_optimizer(
    model: torch.nn.Module,
    config: Any,
    *,
    strict: bool = False,
) -> torch.optim.Optimizer:
    optimizer_cfg = _cfg_get(config, "optimizer")
    optimizer_type = str(_cfg_get(optimizer_cfg, "type", "adam")).strip().lower()
    lr = float(_cfg_get(optimizer_cfg, "lr"))
    eps = float(_cfg_get(optimizer_cfg, "eps"))
    weight_decay = float(_cfg_get(optimizer_cfg, "weight_decay", 0.0))

    if optimizer_type == "adam":
        return torch.optim.Adam(
            list(model.parameters()),
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
    if optimizer_type != "adamw":
        raise ValueError(f"unsupported optimizer.type={optimizer_type!r}")

    betas_cfg = _cfg_get(optimizer_cfg, "betas", [0.9, 0.999])
    if len(betas_cfg) != 2:
        raise ValueError("optimizer.betas must have length 2")
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    filter_frozen = bool(_cfg_get(optimizer_cfg, "filter_frozen", True))
    no_decay_cfg = _parse_no_decay_cfg(optimizer_cfg)
    groups_cfg = _as_dict(_cfg_get(optimizer_cfg, "groups", {}))
    logical_order = _group_order(groups_cfg)

    if strict and "default" not in groups_cfg:
        raise ValueError("production requires optimizer.groups.default")

    dino_cfg = _cfg_get(_cfg_get(_cfg_get(config, "model"), "feature_extractor", {}), "dino", {})
    dino_frozen = bool(_cfg_get(dino_cfg, "freeze", False))
    dino_prefixes: List[str] = []
    if "dino" in groups_cfg:
        dino_match = _cfg_get(groups_cfg["dino"], "match", {})
        dino_prefixes = [str(x) for x in list(_cfg_get(dino_match, "prefixes", []))]

    logical_assignments: Dict[str, str] = {}
    repeated_matches: List[str] = []
    unassigned_trainable: List[str] = []
    frozen_dino_count = 0
    num_trainable = 0
    num_frozen = 0

    tmp_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _touch_bucket(
        logical_name: str,
        split: str,
        *,
        group_lr: float,
        group_wd: float,
    ) -> Dict[str, Any]:
        key = (logical_name, split)
        if key not in tmp_groups:
            bucket_name = f"{logical_name}/{split}"
            tmp_groups[key] = {
                "params": [],
                "param_names": [],
                "name": bucket_name,
                "lr": float(group_lr),
                "weight_decay": float(group_wd),
                "betas": betas,
                "eps": float(eps),
                "logical_name": str(logical_name),
            }
        return tmp_groups[key]

    named_params = list(model.named_parameters())
    for name, p in named_params:
        if p.requires_grad:
            num_trainable += int(p.numel())
        else:
            num_frozen += int(p.numel())
            if dino_prefixes and any(_prefix_match(name, px) for px in dino_prefixes):
                frozen_dino_count += int(p.numel())

        if filter_frozen and not p.requires_grad:
            continue
        if not p.requires_grad:
            continue

        matched: List[str] = []
        for logical_name in logical_order:
            if logical_name == "default":
                continue
            gcfg = groups_cfg.get(logical_name)
            if _match_group(name, _cfg_get(gcfg, "match", {})):
                matched.append(str(logical_name))

        if len(matched) > 1:
            repeated_matches.append(name)
            if strict:
                continue
        if len(matched) == 0:
            if "default" in groups_cfg:
                logical_name = "default"
            else:
                logical_name = ""
                unassigned_trainable.append(name)
                if strict:
                    continue
        else:
            logical_name = matched[0]

        if strict and name in logical_assignments:
            repeated_matches.append(name)
            continue
        logical_assignments[name] = logical_name

        group_cfg = groups_cfg.get(logical_name, {})
        group_lr = float(_cfg_get(group_cfg, "lr", lr))
        group_wd = float(_cfg_get(group_cfg, "weight_decay", weight_decay))
        use_no_decay = _use_no_weight_decay(name, p, no_decay_cfg)
        split = "no_decay" if use_no_decay else "decay"
        effective_wd = 0.0 if use_no_decay else group_wd
        bucket = _touch_bucket(
            logical_name=logical_name,
            split=split,
            group_lr=group_lr,
            group_wd=effective_wd,
        )
        bucket["params"].append(p)
        bucket["param_names"].append(name)

    if strict and repeated_matches:
        preview = repeated_matches[:5]
        raise ValueError(f"optimizer logical group repeated matches: {preview}")
    if strict and unassigned_trainable:
        preview = unassigned_trainable[:5]
        raise ValueError(f"optimizer has unassigned trainable params: {preview}")
    if strict and dino_frozen and len(dino_prefixes) > 0:
        leaked = [
            n
            for n, _ in named_params
            if n in logical_assignments and logical_assignments.get(n) == "dino"
        ]
        if leaked:
            preview = leaked[:5]
            raise ValueError(f"dino.freeze=true but dino params assigned to optimizer: {preview}")

    param_groups: List[Dict[str, Any]] = []
    logical_counts: Dict[str, int] = {}
    for logical_name in logical_order:
        for split in ("decay", "no_decay"):
            bucket = tmp_groups.get((logical_name, split))
            if bucket is None:
                continue
            if len(bucket["params"]) == 0:
                continue
            param_groups.append(bucket)
            logical_counts[logical_name] = logical_counts.get(logical_name, 0) + len(bucket["param_names"])

    if len(param_groups) == 0:
        raise ValueError("optimizer has no trainable parameter groups")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    meta = {
        "num_trainable_params": int(num_trainable),
        "num_frozen_params": int(num_frozen),
        "frozen_dino_params": int(frozen_dino_count),
        "unassigned_trainable_params": int(len(unassigned_trainable)),
        "unassigned_trainable_param_names": list(unassigned_trainable),
        "logical_group_counts": logical_counts,
        "group_signature": _build_meta_signature(param_groups),
    }
    setattr(optimizer, "_streetforward_meta", meta)
    return optimizer


def build_streetforward_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
    *,
    start_step: int = 0,
    strict: bool = False,
) -> Optional[StreetForwardWarmupCosineLR]:
    lr_cfg = _cfg_get(config, "lr_scheduler", {})
    enable = bool(_cfg_get(lr_cfg, "enable", False))
    if not enable:
        return None
    sched_type = str(_cfg_get(lr_cfg, "type", "")).strip().lower()
    if strict and sched_type not in ("cosine", "warmup_cosine"):
        raise ValueError("production requires lr_scheduler.type in {'cosine', 'warmup_cosine'}")
    if sched_type == "warmup_cosine":
        _ = _cfg_get(lr_cfg, "warmup_steps")
        _ = _cfg_get(lr_cfg, "total_steps")
        _ = _cfg_get(lr_cfg, "min_lr_ratio")
        _ = _cfg_get(lr_cfg, "warmup_start_ratio")
        sched_cfg = lr_cfg
    elif sched_type == "cosine":
        _ = _cfg_get(lr_cfg, "total_steps")
        _ = _cfg_get(lr_cfg, "min_lr_ratio")
        sched_cfg = _as_dict(lr_cfg)
        sched_cfg["warmup_steps"] = 0
        sched_cfg["warmup_start_ratio"] = 1.0
    else:
        raise ValueError(f"unsupported lr_scheduler.type={sched_type!r}")
    return StreetForwardWarmupCosineLR(optimizer=optimizer, cfg=sched_cfg, start_step=start_step)
