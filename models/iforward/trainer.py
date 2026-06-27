from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .model import IForwardModel, IForwardRolloutOutput
from .state import IForwardState
from .utils import cfg_get


class IForwardTrainer(nn.Module):
    def __init__(
        self,
        config: Any,
        device: torch.device,
        *,
        model: Optional[IForwardModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.model = model if model is not None else IForwardModel(config=config, device=device)
        self._optimizer_group_names: List[str] = []
        self._optimizer_group_param_ids: Dict[str, List[int]] = {}
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer(config)
        self._apply_trainability_schedule(0)
        self._state_cache: Dict[Tuple[int, int, int], IForwardState] = {}
        self._random_window_metric_cache: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}

    @staticmethod
    def _named_params(module: Optional[nn.Module], prefix: str = "") -> List[Tuple[str, nn.Parameter]]:
        if module is None:
            return []
        out: List[Tuple[str, nn.Parameter]] = []
        for name, param in module.named_parameters(recurse=True):
            out.append((f"{prefix}.{name}" if prefix else str(name), param))
        return out

    @staticmethod
    def _dedupe_params(named_params: Sequence[Tuple[str, nn.Parameter]]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen = set()
        for _, param in named_params:
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(param)
        return params

    def _phase_a_runtime(self) -> Optional[nn.Module]:
        runtime = getattr(self.model, "phase_a_runtime", None)
        if isinstance(runtime, nn.Module):
            return runtime
        bridge = getattr(self.model, "bridge", None)
        runtime = getattr(bridge, "runtime", None)
        return runtime if isinstance(runtime, nn.Module) else None

    def _stage6_posterior_updater(self) -> Optional[nn.Module]:
        runtime = self._phase_a_runtime()
        updater = getattr(runtime, "stage6_posterior_updater", None) if runtime is not None else None
        return updater if isinstance(updater, nn.Module) else None

    def _stage6_struct_decoder(self) -> Optional[nn.Module]:
        runtime = self._phase_a_runtime()
        decoder = getattr(runtime, "stage6_struct_event_decoder", None) if runtime is not None else None
        return decoder if isinstance(decoder, nn.Module) else None

    def _biggs_child_decoder(self) -> Optional[nn.Module]:
        runtime = self._phase_a_runtime()
        decoder = getattr(runtime, "biggs_child_decoder", None) if runtime is not None else None
        return decoder if isinstance(decoder, nn.Module) else None

    def _is_v6_point_mamba_xcpe(self) -> bool:
        if bool(getattr(self.model, "is_v6_point_mamba_xcpe", False)):
            return True
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        return str(cfg_get(iforward_cfg, "version", "")) == "v6_point_mamba_xcpe"

    def _is_stage2_0_biggs_parent_lifting(self) -> bool:
        if bool(getattr(self.model, "is_stage2_0_biggs_parent_lifting", False)):
            return True
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        return str(cfg_get(iforward_cfg, "version", "")) in {
            "stage2_0_biggs_parent_lifting",
            "stage2_0_biggs_cuda_exact_diagonal_projector",
            "stage2_0_biggs_incremental_whdd",
            "stage2_0_biggs_compact16_residualonly",
            "stage2_0_biggs_grld_dinov2base_concat48",
            "stage2_0_fwhr_lift_grld_dinov2base",
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        }

    def _is_stage2_1_parent_temporal(self) -> bool:
        if bool(getattr(self.model, "is_stage2_1_parent_temporal", False)) or bool(
            getattr(self.model, "is_stage2_2_parent_temporal", False)
        ):
            return True
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        return str(cfg_get(iforward_cfg, "version", "")) in {
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        }

    def _is_v3_gru_history_gate(self) -> bool:
        if bool(getattr(self.model, "is_v3_gru_history_gate", False)):
            return True
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        return str(cfg_get(iforward_cfg, "version", "")) == "v3_gru_history_gate"

    def _measurement_frontend_params(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        runtime = self._phase_a_runtime()
        if runtime is None:
            return {
                "stage6_measurement_frontend_residual_unet": [],
                "stage6_measurement_frontend_fusion_neck": [],
                "stage6_measurement_frontend": [],
            }
        trainable_names = set(str(x) for x in getattr(runtime, "stage6_measurement_trainable_param_names", set()))
        if not trainable_names:
            return {
                "stage6_measurement_frontend_residual_unet": [],
                "stage6_measurement_frontend_fusion_neck": [],
                "stage6_measurement_frontend": [],
            }

        named = [(str(name), param) for name, param in runtime.named_parameters(recurse=True) if str(name) in trainable_names]
        residual_prefixes = (
            "image_feature_extractor.residual",
            "image_feature_extractor.residual_unet",
            "image_feature_extractor.detail_head",
        )
        fusion_prefixes = (
            "image_feature_extractor.fusion",
            "image_feature_extractor.fusion_neck",
        )
        residual = [(name, param) for name, param in named if any(name.startswith(prefix) for prefix in residual_prefixes)]
        residual_ids = {id(param) for _, param in residual}
        fusion = [
            (name, param)
            for name, param in named
            if any(name.startswith(prefix) for prefix in fusion_prefixes) and id(param) not in residual_ids
        ]
        assigned_ids = {id(param) for _, param in residual + fusion}
        other = [(name, param) for name, param in named if id(param) not in assigned_ids]
        return {
            "stage6_measurement_frontend_residual_unet": residual,
            "stage6_measurement_frontend_fusion_neck": fusion,
            "stage6_measurement_frontend": other,
        }

    def _group_param_lists(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        updater = self._stage6_posterior_updater()
        adapter = getattr(updater, "vsm_ctx_adapter", None) if updater is not None else None
        adapter_named = self._named_params(
            adapter if isinstance(adapter, nn.Module) else None,
            "stage6_posterior_updater.vsm_ctx_adapter",
        )
        adapter_ids = {id(param) for _, param in adapter_named}
        updater_base = [
            (f"stage6_posterior_updater.{name}", param)
            for name, param in self._named_params(updater, "")
            if id(param) not in adapter_ids
        ]
        struct_decoder = self._stage6_struct_decoder()
        if self._is_v3_gru_history_gate():
            groups = {
                "point_gru": self._named_params(getattr(self.model, "point_gru", None), "point_gru"),
                "history_gate": self._named_params(getattr(self.model, "history_gate", None), "history_gate"),
                "vsm_ctx_adapter": adapter_named,
                "stage6_posterior_updater_base": updater_base,
                "stage6_struct_decoder": self._named_params(struct_decoder, "stage6_struct_event_decoder"),
            }
        elif self._is_v6_point_mamba_xcpe():
            groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {
                "point_mamba": self._named_params(getattr(self.model, "point_mamba", None), "point_mamba"),
                "local_conflict_xcpe": self._named_params(getattr(self.model, "local_conflict", None), "local_conflict"),
                "context_adapter": self._named_params(getattr(self.model, "context_adapter", None), "context_adapter"),
                "vsm_ctx_adapter": adapter_named,
                "stage6_posterior_updater_base": updater_base,
                "stage6_struct_decoder": self._named_params(struct_decoder, "stage6_struct_event_decoder"),
            }
        elif self._is_stage2_1_parent_temporal():
            parent_spatial = getattr(self.model, "parent_spatial_backbone", None)
            parent_temporal = getattr(self.model, "parent_temporal_mamba", None)
            parent_token_named: List[Tuple[str, nn.Parameter]] = []
            parent_ptv3_named: List[Tuple[str, nn.Parameter]] = []
            if isinstance(parent_spatial, nn.Module):
                parent_token_named.extend(self._named_params(getattr(parent_spatial, "param_support_codec", None), "parent_spatial_backbone.param_support_codec"))
                parent_token_named.extend(self._named_params(getattr(parent_spatial, "token_builder", None), "parent_spatial_backbone.token_builder"))
                parent_token_named.extend(self._named_params(getattr(parent_spatial, "far_mlp", None), "parent_spatial_backbone.far_mlp"))
                parent_token_named.extend(self._named_params(getattr(parent_spatial, "far_norm", None), "parent_spatial_backbone.far_norm"))
                parent_ptv3_named.extend(self._named_params(getattr(parent_spatial, "near_ptv3", None), "parent_spatial_backbone.near_ptv3"))
            temporal_adapter_named: List[Tuple[str, nn.Parameter]] = []
            temporal_main_named: List[Tuple[str, nn.Parameter]] = []
            if isinstance(parent_temporal, nn.Module):
                all_temporal = self._named_params(parent_temporal, "parent_temporal_mamba")
                temporal_adapter_named = [(n, p) for n, p in all_temporal if ".adapters." in n]
                adapter_param_ids = {id(p) for _, p in temporal_adapter_named}
                temporal_main_named = [(n, p) for n, p in all_temporal if id(p) not in adapter_param_ids]
            groups = {
                "parent_token_builder": parent_token_named,
                "parent_ptv3": parent_ptv3_named,
                "parent_temporal_mamba": temporal_main_named,
                "parent_temporal_adapter": temporal_adapter_named,
                "stage6_posterior_updater_base": updater_base,
            }
            if adapter_named:
                groups["vsm_ctx_adapter"] = adapter_named
        elif self._is_stage2_0_biggs_parent_lifting():
            groups = {
                "stage6_posterior_updater_base": updater_base,
                "stage6_struct_decoder": self._named_params(struct_decoder, "stage6_struct_event_decoder"),
            }
            if adapter_named:
                groups["vsm_ctx_adapter"] = adapter_named
        else:
            memory = getattr(self.model, "memory", None)
            memory_named = self._named_params(memory if isinstance(memory, nn.Module) else None, "memory")
            memory_main: List[Tuple[str, nn.Parameter]] = []
            memory_fuse: List[Tuple[str, nn.Parameter]] = []
            for name, param in memory_named:
                if ".fuse." in name:
                    memory_fuse.append((name, param))
                else:
                    memory_main.append((name, param))
            groups = {
                "memory": memory_main,
                "memory_fuse": memory_fuse,
                "vsm_ctx_adapter": adapter_named,
                "stage6_posterior_updater_base": updater_base,
                "stage6_struct_decoder": self._named_params(struct_decoder, "stage6_struct_event_decoder"),
            }
        biggs_named = self._named_params(self._biggs_child_decoder(), "biggs_child_decoder")
        if biggs_named:
            groups["biggs_child_decoder"] = biggs_named
        measurement_groups = self._measurement_frontend_params()
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        trainability = cfg_get(iforward_cfg, "trainability", {}) or {}
        train_measurement = bool(cfg_get(trainability, "train_measurement_frontend", False))
        if train_measurement and not any(len(params) > 0 for params in measurement_groups.values()):
            raise ValueError(
                "IForward trainability.train_measurement_frontend=true but Stage6 runtime has no "
                "measurement frontend trainable params. Check phase_a_mode/from_scratch and base_measurement flags."
            )
        for group_name, named_params in measurement_groups.items():
            if named_params:
                groups[str(group_name)] = list(named_params)
        seen: Dict[int, str] = {}
        for group_name, named_params in groups.items():
            for name, param in named_params:
                pid = id(param)
                if pid in seen:
                    raise ValueError(
                        "IForward optimizer parameter appears in multiple groups: "
                        f"{name} in {group_name} already assigned to {seen[pid]}"
                    )
                seen[pid] = str(group_name)
        missing = [str(name) for name, named_params in groups.items() if len(named_params) == 0]
        if missing:
            raise ValueError(f"IForward optimizer group(s) have no parameters: {missing}")
        return groups

    def _lr_for_group(self, config: Any, group_name: str, default_lr: float) -> float:
        opt_cfg = cfg_get(config, "optimizer", {}) or {}
        lr_cfg = cfg_get(opt_cfg, "lr", 1.0e-4)
        if isinstance(lr_cfg, (float, int)):
            return float(lr_cfg)
        fallback = float(cfg_get(lr_cfg, "default", default_lr))
        defaults = {
            "memory": fallback,
            "point_gru": float(cfg_get(lr_cfg, "point_gru", cfg_get(lr_cfg, "memory", fallback))),
            "history_gate": float(cfg_get(lr_cfg, "history_gate", fallback)),
            "point_mamba": float(cfg_get(lr_cfg, "point_mamba", cfg_get(lr_cfg, "memory", fallback))),
            "local_conflict_xcpe": float(cfg_get(lr_cfg, "local_conflict_xcpe", fallback)),
            "context_adapter": float(cfg_get(lr_cfg, "context_adapter", fallback)),
            "memory_fuse": float(cfg_get(lr_cfg, "memory_fuse", fallback)),
            "vsm_ctx_adapter": float(cfg_get(lr_cfg, "vsm_ctx_adapter", 2.0e-4)),
            "stage6_posterior_updater_base": float(cfg_get(lr_cfg, "stage6_posterior_updater_base", 1.0e-5)),
            "stage6_struct_decoder": float(cfg_get(lr_cfg, "stage6_struct_decoder", 0.0)),
            "biggs_child_decoder": float(cfg_get(lr_cfg, "biggs_child_decoder", fallback)),
            "parent_token_builder": float(cfg_get(lr_cfg, "parent_token_builder", fallback)),
            "parent_ptv3": float(cfg_get(lr_cfg, "parent_ptv3", fallback)),
            "parent_temporal_mamba": float(cfg_get(lr_cfg, "parent_temporal_mamba", fallback)),
            "parent_temporal_adapter": float(cfg_get(lr_cfg, "parent_temporal_adapter", fallback)),
            "stage6_measurement_frontend_residual_unet": float(
                cfg_get(lr_cfg, "stage6_measurement_frontend_residual_unet", cfg_get(lr_cfg, "measurement_frontend", fallback))
            ),
            "stage6_measurement_frontend_fusion_neck": float(
                cfg_get(lr_cfg, "stage6_measurement_frontend_fusion_neck", cfg_get(lr_cfg, "measurement_frontend", fallback))
            ),
            "stage6_measurement_frontend": float(cfg_get(lr_cfg, "measurement_frontend", fallback)),
        }
        return float(cfg_get(lr_cfg, group_name, defaults.get(str(group_name), fallback)))

    def _set_all_model_requires_grad(self, value: bool) -> None:
        for param in self.model.parameters():
            param.requires_grad_(bool(value))

    def _set_group_requires_grad(self, group_name: str, value: bool) -> None:
        ids = set(int(x) for x in self._optimizer_group_param_ids.get(str(group_name), []))
        if not ids:
            return
        for param in self.model.parameters():
            if id(param) in ids:
                param.requires_grad_(bool(value))

    def _apply_trainability_schedule(self, global_step: int) -> None:
        if not self._optimizer_group_param_ids:
            return
        iforward_cfg = cfg_get(cfg_get(self.config, "model", {}) or {}, "iforward", {}) or {}
        trainability = cfg_get(iforward_cfg, "trainability", {}) or {}
        unfreeze_updater_step = int(cfg_get(trainability, "unfreeze_updater_base_after_step", 1000))
        train_struct = bool(cfg_get(trainability, "train_stage6_struct_decoder", False))
        unfreeze_struct_step = int(cfg_get(trainability, "unfreeze_struct_decoder_after_step", 10**12))
        train_measurement = bool(cfg_get(trainability, "train_measurement_frontend", False))
        train_biggs_child_decoder = bool(cfg_get(trainability, "train_biggs_child_decoder", True))

        self._set_all_model_requires_grad(False)
        if self._is_v3_gru_history_gate():
            self._set_group_requires_grad(
                "point_gru",
                bool(cfg_get(trainability, "train_point_gru", cfg_get(trainability, "train_memory", True))),
            )
            self._set_group_requires_grad(
                "history_gate",
                bool(cfg_get(trainability, "train_history_gate", True)),
            )
        elif self._is_v6_point_mamba_xcpe():
            self._set_group_requires_grad(
                "point_mamba",
                bool(cfg_get(trainability, "train_point_mamba", cfg_get(trainability, "train_memory", True))),
            )
            self._set_group_requires_grad(
                "local_conflict_xcpe",
                bool(cfg_get(trainability, "train_local_conflict_xcpe", True)),
            )
            self._set_group_requires_grad(
                "context_adapter",
                bool(cfg_get(trainability, "train_context_adapter", True)),
            )
        elif self._is_stage2_1_parent_temporal():
            self._set_group_requires_grad("parent_token_builder", bool(cfg_get(trainability, "train_parent_token_builder", True)))
            self._set_group_requires_grad("parent_ptv3", bool(cfg_get(trainability, "train_parent_ptv3", True)))
            self._set_group_requires_grad("parent_temporal_mamba", bool(cfg_get(trainability, "train_parent_temporal_mamba", True)))
            self._set_group_requires_grad("parent_temporal_adapter", bool(cfg_get(trainability, "train_parent_temporal_adapter", True)))
        else:
            self._set_group_requires_grad("memory", bool(cfg_get(trainability, "train_memory", True)))
            self._set_group_requires_grad("memory_fuse", bool(cfg_get(trainability, "train_memory_fuse", True)))
        self._set_group_requires_grad("vsm_ctx_adapter", bool(cfg_get(trainability, "train_vsm_ctx_adapter", True)))
        updater_train = bool(int(global_step) >= int(unfreeze_updater_step))
        self._set_group_requires_grad("stage6_posterior_updater_base", updater_train)
        struct_train = bool(train_struct and int(global_step) >= int(unfreeze_struct_step))
        self._set_group_requires_grad("stage6_struct_decoder", struct_train)
        self._set_group_requires_grad("biggs_child_decoder", train_biggs_child_decoder)
        for group_name in (
            "stage6_measurement_frontend_residual_unet",
            "stage6_measurement_frontend_fusion_neck",
            "stage6_measurement_frontend",
        ):
            self._set_group_requires_grad(group_name, train_measurement)

        base_lrs = {name: self._lr_for_group(self.config, name, 1.0e-4) for name in self._optimizer_group_names}
        for group in getattr(self, "optimizer", None).param_groups if getattr(self, "optimizer", None) is not None else []:
            name = str(group.get("name", group.get("logical_name", "")))
            lr = float(base_lrs.get(name, group.get("lr", 0.0)))
            if name == "stage6_posterior_updater_base" and not updater_train:
                lr = 0.0
            if name == "stage6_struct_decoder" and not struct_train:
                lr = 0.0
            if name == "biggs_child_decoder" and not train_biggs_child_decoder:
                lr = 0.0
            if name.startswith("stage6_measurement_frontend") and not train_measurement:
                lr = 0.0
            group["lr"] = float(lr)

    def _build_optimizer(self, config: Any) -> torch.optim.Optimizer:
        opt_cfg = cfg_get(config, "optimizer", {}) or {}
        lr_cfg = cfg_get(opt_cfg, "lr", 1.0e-4)
        lr = float(cfg_get(lr_cfg, "default", 1.0e-4) if hasattr(lr_cfg, "get") or isinstance(lr_cfg, dict) else lr_cfg)
        weight_decay = float(cfg_get(opt_cfg, "weight_decay", 0.0))
        betas = tuple(float(x) for x in list(cfg_get(opt_cfg, "betas", [0.9, 0.95]) or [0.9, 0.95]))
        eps = float(cfg_get(opt_cfg, "eps", 1.0e-8))
        named_groups = self._group_param_lists()
        self._set_all_model_requires_grad(False)
        self._optimizer_group_names = list(named_groups.keys())
        param_groups: List[Dict[str, Any]] = []
        for group_name, named_params in named_groups.items():
            params = self._dedupe_params(named_params)
            self._optimizer_group_param_ids[str(group_name)] = [id(param) for param in params]
            group_lr = self._lr_for_group(config, str(group_name), lr)
            param_groups.append(
                {
                    "params": params,
                    "lr": float(group_lr),
                    "weight_decay": weight_decay,
                    "name": str(group_name),
                    "logical_name": str(group_name),
                    "param_names": [str(name) for name, _ in named_params],
                }
            )
        opt_type = str(cfg_get(opt_cfg, "type", "adamw")).lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
        if opt_type == "adam":
            return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
        raise ValueError(f"IForward unsupported optimizer.type={opt_type!r}")

    def forward_rollout(self, *args: Any, **kwargs: Any) -> IForwardRolloutOutput:
        return self.model.forward_rollout(*args, **kwargs)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.forward(batch)

    @staticmethod
    def _grad_norm(parameters: Any) -> torch.Tensor:
        total = None
        ref = None
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            ref = grad
            value = grad.pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            if ref is not None:
                return ref.new_tensor(0.0)
            return torch.tensor(0.0)
        return torch.sqrt(total.clamp_min(0.0))

    @staticmethod
    def _param_count(parameters: Sequence[nn.Parameter]) -> int:
        return int(sum(int(param.numel()) for param in parameters))

    @staticmethod
    def _trainable_param_count(parameters: Sequence[nn.Parameter]) -> int:
        return int(sum(int(param.numel()) for param in parameters if bool(param.requires_grad)))

    def _optimizer_group_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for group in getattr(self.optimizer, "param_groups", []):
            name = str(group.get("name", group.get("logical_name", "group")))
            params = [param for param in list(group.get("params", []) or []) if isinstance(param, nn.Parameter)]
            grad_norm = self._grad_norm(params)
            metrics[f"iforward/optimizer/{name}/lr"] = float(group.get("lr", 0.0))
            metrics[f"iforward/optimizer/{name}/param_count"] = float(self._param_count(params))
            metrics[f"iforward/optimizer/{name}/trainable_param_count"] = float(self._trainable_param_count(params))
            metrics[f"iforward/grad/{name}"] = float(grad_norm.detach().item())
        return metrics

    def _adapter_metrics(self) -> Dict[str, float]:
        updater = self._stage6_posterior_updater()
        adapter = getattr(updater, "vsm_ctx_adapter", None) if updater is not None else None
        if not isinstance(adapter, nn.Module):
            return {}
        out: Dict[str, float] = {}
        weight = getattr(adapter, "weight", None)
        bias = getattr(adapter, "bias", None)
        if torch.is_tensor(weight):
            out["iforward/adapter/vsm_ctx_adapter_weight_norm"] = float(weight.detach().norm().item())
            if weight.grad is not None:
                out["iforward/adapter/vsm_ctx_adapter_weight_grad_norm"] = float(weight.grad.detach().norm().item())
        if torch.is_tensor(bias):
            out["iforward/adapter/vsm_ctx_adapter_bias_norm"] = float(bias.detach().norm().item())
            if bias.grad is not None:
                out["iforward/adapter/vsm_ctx_adapter_bias_grad_norm"] = float(bias.grad.detach().norm().item())
        params = list(adapter.parameters())
        if params:
            out["iforward/adapter/vsm_ctx_adapter_grad_norm"] = float(self._grad_norm(params).detach().item())
        return out

    def _cache_key_from_output(self, out: IForwardRolloutOutput) -> Tuple[int, int, int]:
        return tuple(out.resolved.cache_key)

    def _clear_state_cache_for_new_episode(self) -> int:
        removed = int(len(self._state_cache))
        self._state_cache.clear()
        self._random_window_metric_cache.clear()
        return removed

    @staticmethod
    def _finite_metric(value: Any) -> Optional[float]:
        if not isinstance(value, (int, float)):
            return None
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None

    def _random_window_revisit_metrics(self, out: IForwardRolloutOutput) -> Dict[str, float]:
        resolved = out.resolved
        if str(resolved.scheduler_version) not in {"random_window_v1", "iforward_v3_random_window"} or int(resolved.window_hash) < 0:
            return {}
        key = (
            int(resolved.scene_id),
            int(resolved.segment_id),
            int(resolved.episode_id),
            int(resolved.window_hash),
        )
        current = {
            "current": self._finite_metric(out.stats.get("current_psnr", out.stats.get("current_latest_psnr"))),
            "history": self._finite_metric(out.stats.get("history_psnr", out.stats.get("in_rollout_history_psnr"))),
            "nearby": self._finite_metric(out.stats.get("nearby_psnr")),
        }
        previous = self._random_window_metric_cache.get(key)
        self._random_window_metric_cache[key] = {
            name: float(value)
            for name, value in current.items()
            if value is not None
        }
        if not previous:
            return {}
        metrics: Dict[str, float] = {}
        for name, value in current.items():
            if value is None or name not in previous:
                continue
            metrics[f"iforward/revisit/{name}_psnr_delta"] = float(value) - float(previous[name])
        return metrics

    @staticmethod
    def _sync_cuda(enabled: bool) -> None:
        if bool(enabled) and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _cuda_memory_snapshot() -> Dict[str, int]:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_bytes = int(free_bytes)
            total_bytes = int(total_bytes)
            driver_used = int(total_bytes - free_bytes)
        except Exception:
            free_bytes = 0
            total_bytes = 0
            driver_used = 0
        return {
            "allocated": allocated,
            "reserved": reserved,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "driver_used": driver_used,
            "driver_free": free_bytes,
            "driver_total": total_bytes,
        }

    @staticmethod
    def _record_cuda_memory_snapshot(
        metrics: Dict[str, float],
        snapshots: Dict[str, Dict[str, int]],
        label: str,
        *,
        baseline: Optional[Dict[str, int]],
    ) -> None:
        snap = IForwardTrainer._cuda_memory_snapshot()
        if not snap:
            return
        snapshots[str(label)] = snap
        scale = 1024.0 * 1024.0
        prefix = f"perf/cuda/{label}"
        metrics[f"{prefix}_allocated_mb"] = float(snap["allocated"] / scale)
        metrics[f"{prefix}_reserved_mb"] = float(snap["reserved"] / scale)
        metrics[f"{prefix}_peak_allocated_mb"] = float(snap["peak_allocated"] / scale)
        metrics[f"{prefix}_peak_reserved_mb"] = float(snap["peak_reserved"] / scale)
        metrics[f"{prefix}_driver_used_mb"] = float(snap["driver_used"] / scale)
        metrics[f"{prefix}_driver_free_mb"] = float(snap["driver_free"] / scale)
        metrics[f"{prefix}_driver_total_mb"] = float(snap["driver_total"] / scale)
        metrics[f"{prefix}_reserved_minus_allocated_mb"] = float((snap["reserved"] - snap["allocated"]) / scale)
        metrics[f"{prefix}_driver_used_minus_reserved_mb"] = float((snap["driver_used"] - snap["reserved"]) / scale)
        if baseline:
            metrics[f"{prefix}_allocated_delta_mb"] = float((snap["allocated"] - baseline["allocated"]) / scale)
            metrics[f"{prefix}_reserved_delta_mb"] = float((snap["reserved"] - baseline["reserved"]) / scale)
            metrics[f"{prefix}_driver_used_delta_mb"] = float((snap["driver_used"] - baseline["driver_used"]) / scale)

    @staticmethod
    def _record_cuda_phase_delta(
        metrics: Dict[str, float],
        snapshots: Dict[str, Dict[str, int]],
        phase: str,
        start_label: str,
        end_label: str,
    ) -> None:
        start = snapshots.get(str(start_label))
        end = snapshots.get(str(end_label))
        if not start or not end:
            return
        scale = 1024.0 * 1024.0
        prefix = f"perf/cuda/{phase}"
        for name in ("allocated", "reserved", "driver_used"):
            metrics[f"{prefix}_{name}_delta_mb"] = float((end[name] - start[name]) / scale)

    def _reset_bridge_runtime_node_state(self) -> Dict[str, int]:
        bridge = getattr(self.model, "bridge", None)
        reset = getattr(bridge, "reset_runtime_node_state", None)
        if not callable(reset):
            return {}
        return dict(reset())

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        profile_cuda_memory: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[Any] = None,
        ablation: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = (scheduler_node_sync, runtime_policy)
        profile_cuda = bool(profile_phase_timing or sync_cuda_timing)
        profile_memory = bool(profile_cuda_memory and torch.cuda.is_available())
        timings: Dict[str, float] = {}
        cuda_memory_metrics: Dict[str, float] = {}
        cuda_memory_snapshots: Dict[str, Dict[str, int]] = {}
        cuda_memory_baseline: Optional[Dict[str, int]] = None
        if profile_memory:
            cuda_memory_baseline = self._cuda_memory_snapshot()
            if cuda_memory_baseline:
                cuda_memory_snapshots["start"] = cuda_memory_baseline
                self._record_cuda_memory_snapshot(
                    cuda_memory_metrics,
                    cuda_memory_snapshots,
                    "start",
                    baseline=cuda_memory_baseline,
                )
        batch = dict(batch)
        batch["global_step"] = int(step or 0)
        self._apply_trainability_schedule(int(step or 0))
        t0 = time.perf_counter()
        resolved = self.model.resolver.resolve(batch)
        timings["resolve_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_resolve",
                baseline=cuda_memory_baseline,
            )
        t0 = time.perf_counter()
        key = tuple(resolved.cache_key)
        runtime_reset_before: Dict[str, int] = {}
        stale_cache_entries_cleared = 0
        if bool(resolved.reset_scene_state_before_rollout):
            stale_cache_entries_cleared = self._clear_state_cache_for_new_episode()
            runtime_reset_before = self._reset_bridge_runtime_node_state()
        carried = self._state_cache.get(key)
        timings["state_cache_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_state_cache_lookup",
                baseline=cuda_memory_baseline,
            )

        self.train(True)
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        timings["optimizer_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_zero_grad",
                baseline=cuda_memory_baseline,
            )
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        out = self.model.forward_rollout(batch, carried_state=carried, ablation=ablation)
        self._sync_cuda(profile_cuda)
        timings["forward_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_forward",
                baseline=cuda_memory_baseline,
            )
        loss = out.loss
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        loss.backward()
        self._sync_cuda(profile_cuda)
        timings["backward_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_backward",
                baseline=cuda_memory_baseline,
            )
        t0 = time.perf_counter()
        params_with_grad = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        grad_clip_cfg = cfg_get(cfg_get(self.config, "training", {}) or {}, "grad_clip", {}) or {}
        grad_clip_enable = bool(cfg_get(grad_clip_cfg, "enable", False))
        grad_clip_max_norm = float(cfg_get(grad_clip_cfg, "max_norm", 1.0))
        grad_clip_invoked = bool(grad_clip_enable and params_with_grad)
        grad_clip_was_active = False
        grad_clip_scale = 1.0
        if grad_clip_enable and params_with_grad:
            unclipped = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=float(grad_clip_max_norm))
            grad_norm_unclipped = torch.as_tensor(unclipped, device=loss.device, dtype=loss.dtype)
        else:
            grad_norm_unclipped = self._grad_norm(params_with_grad).to(device=loss.device)
        if not torch.isfinite(grad_norm_unclipped).all():
            raise RuntimeError("IForward gradient norm became NaN/Inf.")
        if grad_clip_enable and params_with_grad:
            grad_norm_value = float(grad_norm_unclipped.detach().item())
            if grad_norm_value > float(grad_clip_max_norm):
                grad_clip_was_active = True
                grad_clip_scale = float(grad_clip_max_norm) / max(float(grad_norm_value), 1.0e-12)
            grad_norm_after_clip = grad_norm_unclipped.new_tensor(min(float(grad_norm_value), float(grad_clip_max_norm)))
        else:
            grad_norm_after_clip = grad_norm_unclipped
        if not torch.isfinite(grad_norm_after_clip).all():
            raise RuntimeError("IForward clipped gradient norm became NaN/Inf.")
        group_metrics = self._optimizer_group_metrics()
        adapter_metrics = self._adapter_metrics()
        timings["grad_norm_ms"] = (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_grad_norm",
                baseline=cuda_memory_baseline,
            )
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._sync_cuda(profile_cuda)
        timings["optimizer_ms"] += (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_optimizer",
                baseline=cuda_memory_baseline,
            )

        t0 = time.perf_counter()
        runtime_reset_after: Dict[str, int] = {}
        if bool(out.resolved.carry_scene_state_after_rollout) and not bool(out.resolved.episode_end_after_rollout):
            self._state_cache[key] = out.next_state.detach_for_next_rollout()
        else:
            self._state_cache.pop(key, None)
            runtime_reset_after = self._reset_bridge_runtime_node_state()
        timings["state_cache_ms"] += (time.perf_counter() - t0) * 1000.0
        if profile_memory:
            self._record_cuda_memory_snapshot(
                cuda_memory_metrics,
                cuda_memory_snapshots,
                "after_state_cache_update",
                baseline=cuda_memory_baseline,
            )

        t0 = time.perf_counter()
        losses = {name: float(value.detach().item()) for name, value in out.losses.items()}
        final = {
            "loss": float(loss.detach().item()),
            "iforward/loss_total": float(loss.detach().item()),
            "iforward/scene_id": int(out.resolved.scene_id),
            "iforward/segment_id": int(out.resolved.segment_id),
            "iforward/episode_id": int(out.resolved.episode_id),
            "iforward/inner_K": float(out.resolved.inner_K),
            "iforward/rollout_id_global": float(out.resolved.rollout_id_global),
            "iforward/rollout_idx_in_episode": float(out.resolved.rollout_idx_in_episode),
            "iforward/rollouts_per_episode": int(out.resolved.rollouts_per_episode),
            "iforward/state_age_rollouts": int(max(int(out.resolved.rollout_idx_in_episode), 0)),
            "iforward/state_age_inner_steps": int(
                out.resolved.steps[0].optimizer_step_idx_in_episode if out.resolved.steps else 0
            ),
            "iforward/reset_scene_state_before_rollout": bool(out.resolved.reset_scene_state_before_rollout),
            "iforward/episode_end_after_rollout": bool(out.resolved.episode_end_after_rollout),
            "iforward/carry_scene_state_after_rollout": bool(out.resolved.carry_scene_state_after_rollout),
            "iforward/state_cache_size": int(len(self._state_cache)),
            "iforward/stale_state_cache_entries_cleared": int(stale_cache_entries_cleared),
            "iforward/grad_norm_total": float(grad_norm_after_clip.detach().item()),
            "iforward/grad_norm_unclipped": float(grad_norm_unclipped.detach().item()),
            "iforward/grad_norm_after_clip": float(grad_norm_after_clip.detach().item()),
            "iforward/grad_clip_max_norm": float(grad_clip_max_norm),
            "iforward/grad_clip_invoked": bool(grad_clip_invoked),
            "iforward/grad_clip_was_active": bool(grad_clip_was_active),
            "iforward/grad_clip_scale": float(grad_clip_scale),
            "iforward/grad_clip_applied": bool(grad_clip_was_active),
            "iforward/runtime_node_state_reset_before": bool(runtime_reset_before),
            "iforward/runtime_node_state_reset_after": bool(runtime_reset_after),
            "num_targets": int(out.stats.get("num_targets", 0)),
            "num_source_views": int(out.stats.get("num_source_views", 0)),
            "num_gaussians_bg": int(out.stats.get("num_gaussians_bg", 0)),
            "num_gaussians_distant": int(out.stats.get("num_gaussians_distant", 0)),
            "num_gaussians_rigid": int(out.stats.get("num_gaussians_rigid", 0)),
            "num_gaussians_sky": int(out.stats.get("num_gaussians_sky", 0)),
            "pred_rgbs": [x.detach().float().cpu() for x in out.pred_rgbs],
            "gt_images": [x.detach().float().cpu() for x in out.gt_images],
            "image_refs": [tuple(int(v) for v in ref) for ref in out.image_refs],
            "image_roles": [str(role) for role in out.image_roles],
        }
        final["iforward/scheduler_version"] = str(out.resolved.scheduler_version)
        final["iforward/window_block_ids"] = [int(x) for x in tuple(out.resolved.window_block_ids)]
        if str(out.resolved.scheduler_version) == "iforward_sequence10_v1":
            meta = dict(getattr(out.resolved, "meta", {}) or {})
            sequence_positions = [int(x) for x in list(meta.get("sequence_positions", []) or [])]
            sequence_keyframes = [int(x) for x in list(meta.get("sequence_keyframe_indices", []) or [])]
            sequence_frames = [int(x) for x in list(meta.get("sequence_source_frame_indices", []) or [])]
            repair_positions = [int(x) for x in list(meta.get("repair_positions", []) or [])]
            final["iforward/sequence10/phase"] = str(meta.get("scheduler_phase", ""))
            final["iforward/sequence10/rollout_phase"] = str(meta.get("rollout_phase", ""))
            final["iforward/sequence10/stride"] = int(meta.get("sequence_stride", 0) or 0)
            final["iforward/sequence10/sequence_id"] = int(meta.get("sequence_id", -1) or -1)
            final["iforward/sequence10/positions"] = sequence_positions
            final["iforward/sequence10/keyframe_ids"] = sequence_keyframes
            final["iforward/sequence10/frame_ids"] = sequence_frames
            final["iforward/sequence10/history_positions"] = [
                int(x) for x in list(meta.get("history_positions", []) or [])
            ]
            final["iforward/sequence10/repair_positions"] = repair_positions
            final["iforward/sequence10/repair_hash"] = int(meta.get("repair_permutation_hash", -1) or -1)
            final["iforward/sequence10/repair_flag"] = bool(str(meta.get("scheduler_phase", "")) == "repair")
            final["iforward/sequence10/temporal_commit_count"] = int(
                sum(1 for step in out.resolved.steps if bool(getattr(step, "temporal_commit", False)))
            )
            final["iforward/sequence10/temporal_read_count"] = int(
                sum(1 for step in out.resolved.steps if bool(getattr(step, "temporal_read", False)))
            )
            final["iforward/sequence10/history_frame_count"] = int(len(list(meta.get("history_positions", []) or [])))
        if str(out.resolved.scheduler_version) == "iforward_stage2_2_stream10_rawframe":
            meta = dict(getattr(out.resolved, "meta", {}) or {})
            stage22_request_meta = dict((dict(meta.get("request_meta", {}) or {})).get("iforward_stage2_2", {}) or {})
            sequence_positions = [int(x) for x in list(meta.get("sequence_positions", []) or [])]
            sequence_keyframes = [int(x) for x in list(meta.get("sequence_keyframe_indices", []) or [])]
            sequence_frames = [
                int(x)
                for x in list(meta.get("sequence_source_frame_indices", stage22_request_meta.get("raw_frame_ids", [])) or [])
            ]
            timestamps = [
                int(x)
                for x in list(meta.get("sequence_timestamps_us", stage22_request_meta.get("timestamps_us", [])) or [])
            ]
            repair_positions = [int(x) for x in list(meta.get("repair_positions", []) or [])]
            final["iforward/stage2_2/protocol"] = str(meta.get("sequence_protocol", ""))
            final["iforward/stage2_2/phase"] = str(meta.get("scheduler_phase", ""))
            final["iforward/stage2_2/rollout_phase"] = str(meta.get("rollout_phase", ""))
            final["iforward/stage2_2/sequence_id"] = int(meta.get("sequence_id", -1) or -1)
            final["iforward/stage2_2/positions"] = sequence_positions
            final["iforward/stage2_2/keyframe_ids"] = sequence_keyframes
            final["iforward/stage2_2/raw_frame_ids"] = sequence_frames
            final["iforward/stage2_2/timestamps_us"] = timestamps
            final["iforward/stage2_2/frame_gaps"] = [
                int(getattr(step, "frame_gap", 0)) for step in out.resolved.steps if int(getattr(step, "repeat_idx", 0)) == 0
            ]
            final["iforward/stage2_2/history_positions"] = [
                int(x) for x in list(meta.get("history_positions", []) or [])
            ]
            final["iforward/stage2_2/repair_positions"] = repair_positions
            final["iforward/stage2_2/repair_hash"] = int(meta.get("repair_permutation_hash", -1) or -1)
            final["iforward/stage2_2/repair_flag"] = bool(str(meta.get("scheduler_phase", "")) == "repair")
            final["iforward/stage2_2/index_fingerprint"] = str(
                meta.get("index_fingerprint", stage22_request_meta.get("index_fingerprint", ""))
            )
            final["iforward/stage2_2/temporal_commit_count"] = int(
                sum(1 for step in out.resolved.steps if bool(getattr(step, "temporal_commit", False)))
            )
            final["iforward/stage2_2/temporal_read_count"] = int(
                sum(1 for step in out.resolved.steps if bool(getattr(step, "temporal_read", False)))
            )
            final["iforward/stage2_2/history_frame_count"] = int(len(list(meta.get("history_positions", []) or [])))
            final_supervision = dict(meta.get("final_supervision", {}) or {})
            final["iforward/stage2_2/history_ref_count"] = int(
                final_supervision.get("history_ref_count", meta.get("history_ref_count", 0)) or 0
            )
        for name, value in timings.items():
            final[name] = float(value)
        for prefix, values in (
            ("iforward/runtime_node_state_reset_before", runtime_reset_before),
            ("iforward/runtime_node_state_reset_after", runtime_reset_after),
        ):
            for name, value in values.items():
                final[f"{prefix}/{name}"] = int(value)
        for name, value in losses.items():
            final[f"iforward/loss_{name}"] = float(value)
        final.update(group_metrics)
        final.update(adapter_metrics)
        final.update(cuda_memory_metrics)
        if profile_memory:
            self._record_cuda_phase_delta(
                final,
                cuda_memory_snapshots,
                "forward",
                "after_zero_grad",
                "after_forward",
            )
            self._record_cuda_phase_delta(
                final,
                cuda_memory_snapshots,
                "backward",
                "after_forward",
                "after_backward",
            )
            self._record_cuda_phase_delta(
                final,
                cuda_memory_snapshots,
                "grad_norm",
                "after_backward",
                "after_grad_norm",
            )
            self._record_cuda_phase_delta(
                final,
                cuda_memory_snapshots,
                "optimizer_step",
                "after_grad_norm",
                "after_optimizer",
            )
            self._record_cuda_phase_delta(
                final,
                cuda_memory_snapshots,
                "state_cache_update",
                "after_optimizer",
                "after_state_cache_update",
            )
        final.update(self._random_window_revisit_metrics(out))
        for name, value in out.stats.items():
            out_name = str(name) if str(name).startswith("iforward/") else f"iforward/{name}"
            if isinstance(value, bool):
                final[out_name] = bool(value)
            elif isinstance(value, int):
                final[out_name] = int(value)
            elif isinstance(value, float) and math.isfinite(float(value)):
                final[out_name] = float(value)
            elif isinstance(value, str):
                final[out_name] = value
        memory_tokens = out.stats.get("memory_tokens")
        if isinstance(memory_tokens, dict):
            for name, value in memory_tokens.items():
                if isinstance(value, bool):
                    final[f"iforward/memory_tokens/{name}"] = bool(value)
                elif isinstance(value, int):
                    final[f"iforward/memory_tokens/{name}"] = int(value)
                elif isinstance(value, float) and math.isfinite(float(value)):
                    final[f"iforward/memory_tokens/{name}"] = float(value)
        model_cfg = cfg_get(self.config, "model", {}) or {}
        iforward_cfg = cfg_get(model_cfg, "iforward", {}) or {}
        debug_cfg = cfg_get(iforward_cfg, "debug", {}) or {}
        log_per_k_metrics = bool(cfg_get(debug_cfg, "log_per_k_metrics", True))
        if log_per_k_metrics:
            for item in out.per_step:
                k = int(item.get("k", 0))
                for name, value in item.items():
                    if name == "k" or not isinstance(value, (int, float)):
                        continue
                    value_f = float(value)
                    if math.isfinite(value_f):
                        final[f"iforward/k{k}/{name}"] = value_f
        final["logging_pack_ms"] = float((time.perf_counter() - t0) * 1000.0)
        if profile_memory:
            self._record_cuda_memory_snapshot(
                final,
                cuda_memory_snapshots,
                "after_logging_pack",
                baseline=cuda_memory_baseline,
            )
        return final

    def reset_iforward_state_cache(self) -> None:
        self._state_cache.clear()

    def load_init_checkpoint_payload(
        self,
        ckpt: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        weights_only: bool = True,
        path: Optional[str] = None,
    ) -> bool:
        loader = getattr(self.model, "load_init_checkpoint_payload", None)
        if callable(loader):
            return bool(loader(ckpt, device=device or self.device, weights_only=weights_only, path=path))
        return False

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "format": "iforward_trainer_extra_state_v1",
            "state_cache": {
                tuple(key): value.detach_for_next_rollout()
                for key, value in self._state_cache.items()
            },
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            self._state_cache = {}
            return
        raw_cache = state.get("state_cache", {})
        if not isinstance(raw_cache, dict):
            self._state_cache = {}
            return
        self._state_cache = {
            tuple(int(x) for x in key): value.detach_for_next_rollout()
            for key, value in raw_cache.items()
        }

    def load_optimizer_state_from_checkpoint(self, payload: Dict[str, Any]) -> bool:
        opt_state = payload.get("optimizer_state_dict")
        if opt_state is None:
            return False
        self.optimizer.load_state_dict(opt_state)
        return True
