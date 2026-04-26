"""
Stage5_3 production training wrapper:
- fast-fail bad-step policy
- AdamW + warmup cosine + AMP scaler
- lightweight runtime reset API for resume
"""

from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Any, Dict, Optional

import torch

from models.streetforward.minimal_trainer_stage4_0 import _backward_to_render_params_bg_rigid_distant
from models.streetforward.minimal_trainer_stage5_3 import MinimalStreetForwardStage5_3
from models.streetforward.production_training_utils import (
    ProductionAmpConfig,
    ProductionBadStepConfig,
    ProductionGradClipConfig,
    ProductionLRSchedulerConfig,
    ProductionOptimizerConfig,
    build_adamw_optimizer,
    build_grad_scaler,
    build_warmup_cosine_scheduler,
    iter_trainable_parameters,
    parse_betas,
    resolve_amp_dtype,
)


class MinimalStreetForwardStage5_3_Production(MinimalStreetForwardStage5_3):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_production_config(config)
        self._init_production_training_config(config)
        super().__init__(config=config, device=device, **kwargs)

    def _validate_production_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if not bool(self._require_key(model_cfg, "production_training", "model")):
            raise ValueError("Stage5_3 production requires model.production_training=true.")
        optimizer_cfg = self._require_key(config, "optimizer", "config")
        _ = self._require_key(optimizer_cfg, "type", "optimizer")
        _ = self._require_key(optimizer_cfg, "default_lr", "optimizer")
        _ = self._require_key(optimizer_cfg, "default_weight_decay", "optimizer")
        _ = self._require_key(optimizer_cfg, "betas", "optimizer")
        _ = self._require_key(optimizer_cfg, "eps", "optimizer")
        training_cfg = self._require_key(config, "training", "config")
        amp_cfg = self._require_key(training_cfg, "amp", "training")
        _ = self._require_key(amp_cfg, "enable", "training.amp")
        _ = self._require_key(amp_cfg, "dtype", "training.amp")
        grad_clip_cfg = self._require_key(training_cfg, "grad_clip", "training")
        _ = self._require_key(grad_clip_cfg, "enable", "training.grad_clip")
        _ = self._require_key(grad_clip_cfg, "max_norm", "training.grad_clip")
        _ = self._require_key(grad_clip_cfg, "norm_type", "training.grad_clip")
        bad_step_cfg = self._require_key(training_cfg, "bad_step", "training")
        _ = self._require_key(bad_step_cfg, "policy", "training.bad_step")
        if str(self._require_key(bad_step_cfg, "policy", "training.bad_step")) != "fast_fail":
            raise ValueError("Stage5_3 production requires training.bad_step.policy=fast_fail.")
        _ = self._require_key(bad_step_cfg, "fail_on_nonfinite_loss", "training.bad_step")
        _ = self._require_key(bad_step_cfg, "fail_on_nonfinite_grad", "training.bad_step")
        _ = self._require_key(bad_step_cfg, "fail_on_amp_overflow", "training.bad_step")
        if not hasattr(bad_step_cfg, "__contains__") or "fail_on_grad_norm_gt" not in bad_step_cfg:
            raise ValueError("Missing required config: training.bad_step.fail_on_grad_norm_gt")
        lr_scheduler_cfg = self._require_key(config, "lr_scheduler", "config")
        if str(self._require_key(lr_scheduler_cfg, "type", "lr_scheduler")) != "warmup_cosine":
            raise ValueError("Stage5_3 production requires lr_scheduler.type=warmup_cosine.")
        _ = self._require_key(lr_scheduler_cfg, "warmup_steps", "lr_scheduler")
        _ = self._require_key(lr_scheduler_cfg, "min_lr_ratio", "lr_scheduler")
        _ = self._require_key(training_cfg, "max_iterations", "training")

    def _init_production_training_config(self, config) -> None:
        optimizer_cfg = self._require_key(config, "optimizer", "config")
        self._prod_optimizer_cfg = ProductionOptimizerConfig(
            lr=float(self._require_key(optimizer_cfg, "default_lr", "optimizer")),
            weight_decay=float(self._require_key(optimizer_cfg, "default_weight_decay", "optimizer")),
            betas=parse_betas(self._require_key(optimizer_cfg, "betas", "optimizer")),
            eps=float(self._require_key(optimizer_cfg, "eps", "optimizer")),
        )
        training_cfg = self._require_key(config, "training", "config")
        amp_cfg = self._require_key(training_cfg, "amp", "training")
        self._prod_amp_cfg = ProductionAmpConfig(
            enable=bool(self._require_key(amp_cfg, "enable", "training.amp")),
            dtype=str(self._require_key(amp_cfg, "dtype", "training.amp")),
        )
        grad_clip_cfg = self._require_key(training_cfg, "grad_clip", "training")
        self._prod_grad_clip_cfg = ProductionGradClipConfig(
            enable=bool(self._require_key(grad_clip_cfg, "enable", "training.grad_clip")),
            max_norm=float(self._require_key(grad_clip_cfg, "max_norm", "training.grad_clip")),
            norm_type=float(self._require_key(grad_clip_cfg, "norm_type", "training.grad_clip")),
        )
        bad_step_cfg = self._require_key(training_cfg, "bad_step", "training")
        self._prod_bad_step_cfg = ProductionBadStepConfig(
            fail_on_nonfinite_loss=bool(self._require_key(bad_step_cfg, "fail_on_nonfinite_loss", "training.bad_step")),
            fail_on_nonfinite_grad=bool(self._require_key(bad_step_cfg, "fail_on_nonfinite_grad", "training.bad_step")),
            fail_on_amp_overflow=bool(self._require_key(bad_step_cfg, "fail_on_amp_overflow", "training.bad_step")),
            fail_on_grad_norm_gt=(
                None
                if bad_step_cfg.get("fail_on_grad_norm_gt", None) is None
                else float(bad_step_cfg.get("fail_on_grad_norm_gt"))
            ),
        )
        lr_scheduler_cfg = self._require_key(config, "lr_scheduler", "config")
        self._prod_lr_scheduler_cfg = ProductionLRSchedulerConfig(
            warmup_steps=int(self._require_key(lr_scheduler_cfg, "warmup_steps", "lr_scheduler")),
            total_steps=int(self._require_key(training_cfg, "max_iterations", "training")),
            min_lr_ratio=float(self._require_key(lr_scheduler_cfg, "min_lr_ratio", "lr_scheduler")),
        )

    def _rebuild_optimizer_after_stage5_modules(self) -> None:
        self.optimizer = build_adamw_optimizer(self, cfg=self._prod_optimizer_cfg)
        self.lr_scheduler = build_warmup_cosine_scheduler(self.optimizer, cfg=self._prod_lr_scheduler_cfg)
        self.grad_scaler = build_grad_scaler(amp_cfg=self._prod_amp_cfg, device=self.device)

    def _assert_finite_loss(self, loss: torch.Tensor, *, step: Optional[int]) -> None:
        if not bool(self._prod_bad_step_cfg.fail_on_nonfinite_loss):
            return
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(
                f"non-finite loss detected at step={step}: value={float(loss.detach().item()) if loss.numel() == 1 else 'tensor'}"
            )

    def _assert_finite_gradients(self, *, step: Optional[int]) -> None:
        if not bool(self._prod_bad_step_cfg.fail_on_nonfinite_grad):
            return
        for p in iter_trainable_parameters(self):
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite gradient detected at step={step}")

    def _compute_total_grad_norm(self) -> float:
        params = list(iter_trainable_parameters(self))
        if len(params) == 0:
            return 0.0
        if bool(self._prod_grad_clip_cfg.enable):
            val = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=float(self._prod_grad_clip_cfg.max_norm),
                norm_type=float(self._prod_grad_clip_cfg.norm_type),
                error_if_nonfinite=False,
            )
            return float(val.item() if torch.is_tensor(val) else val)
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if len(grads) == 0:
            return 0.0
        tot = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        return float(tot.item())

    def clear_runtime_state_for_lightweight_resume(self) -> None:
        # Lightweight resume is not bitwise continuation:
        # keep network/optimizer/scheduler states from checkpoint,
        # but re-initialize runtime caches/node state from upcoming batches.
        self.h_cache_bg.clear()
        self.h_cache_distant.clear()
        self.h_cache_rigid.clear()
        self.stage5_2_history_bg.clear()
        self.stage5_2_history_distant.clear()
        self.stage5_2_history_rigid.clear()
        self.stage5_2_block_support_bg.clear()
        self.stage5_2_block_support_distant.clear()
        self.stage5_2_block_support_rigid.clear()
        self.reset_node_state()

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.train()
        self._perf_acc = {}
        node_state_sync_update = False
        node_state_sync_reset = False
        timing_ms: Dict[str, float] = {"forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}
        amp_enabled = bool(self._prod_amp_cfg.enable and str(self.device.type) == "cuda")
        amp_dtype = resolve_amp_dtype(self._prod_amp_cfg.dtype)

        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if amp_enabled else nullcontext()
        )
        with autocast_ctx:
            out = self.forward(batch)
            loss = out.get("loss")
        if not torch.is_tensor(loss):
            raise ValueError("train_step expects forward() to return tensor loss.")
        self._assert_finite_loss(loss, step=step)
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timing_ms["forward_ms"] = float((t1 - t0) * 1000.0)
        else:
            t1 = time.perf_counter()

        if amp_enabled:
            scale_before = float(self.grad_scaler.get_scale())
            self.grad_scaler.scale(loss).backward()
        else:
            scale_before = 1.0
            loss.backward()
        if out.get("proxies") is not None:
            # IMPORTANT: proxy grads are AMP-scaled at this point when amp_enabled=True.
            # _backward_to_render_params_bg_rigid_distant must only propagate gradients
            # and must not interpret gradient magnitudes.
            _backward_to_render_params_bg_rigid_distant(
                out["render_params"],
                out["proxies"],
                out.get("_render_params_rigid_world"),
                out.get("_proxies_rigid_world"),
                out.get("_render_params_distant"),
                out.get("_proxies_distant"),
                rigid_world_proxy_pairs=out.get("_rigid_world_proxy_pairs"),
            )
        if amp_enabled:
            self.grad_scaler.unscale_(self.optimizer)
        self._assert_finite_gradients(step=step)
        grad_norms = self._compute_branch_grad_norms()
        total_grad_norm = self._compute_total_grad_norm()
        if not torch.isfinite(torch.tensor(total_grad_norm, device=self.device)).all():
            raise FloatingPointError(f"non-finite total grad norm at step={step}: {total_grad_norm}")
        grad_norm_threshold = self._prod_bad_step_cfg.fail_on_grad_norm_gt
        if grad_norm_threshold is not None and float(total_grad_norm) > float(grad_norm_threshold):
            raise FloatingPointError(
                f"grad_norm too large at step={step}: {total_grad_norm} > {grad_norm_threshold}"
            )
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            timing_ms["backward_ms"] = float((t2 - t1) * 1000.0)
        else:
            t2 = time.perf_counter()

        if amp_enabled:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            scale_after = float(self.grad_scaler.get_scale())
            if bool(self._prod_bad_step_cfg.fail_on_amp_overflow) and scale_after < scale_before:
                # Fast-fail semantics: this exception is terminal for the current process.
                raise FloatingPointError(
                    f"AMP overflow detected at step={step}: scale_before={scale_before}, scale_after={scale_after}"
                )
        else:
            self.optimizer.step()
            scale_after = scale_before
        self.lr_scheduler.step()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            timing_ms["optimizer_ms"] = float((t3 - t2) * 1000.0)
        else:
            t3 = time.perf_counter()

        if "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()

        if scheduler_node_sync is not None:
            u_steps = int(scheduler_node_sync["U"])
            seg = int(scheduler_node_sync["segment_local_step"])
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False))
            if u_steps < 1:
                raise ValueError("scheduler_node_sync requires U >= 1 (scheduler time_base.state_write_interval_steps).")
            if seg > 0 and seg % u_steps == 0:
                self._writeback_node_states_from_out(out)
                node_state_sync_update = True
            if reset_after_block:
                self.reset_node_state()
                node_state_sync_reset = True

        num_gaussians_bg = int(out["_node_state_bg"].means.shape[0])
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")
        num_gaussians_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_gaussians_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        num_rigid_valid_src = int(out.get("_num_rigid_valid_src", 0))
        num_rigid_total = int(out.get("_num_rigid_total", num_gaussians_rigid))
        writeback_idx = out.get("_rigid_writeback_idx")
        writeback_count = int(writeback_idx.numel()) if writeback_idx is not None else 0
        writeback_rigid_ratio = float(writeback_count / max(num_rigid_total, 1))
        bg_w_idx = out.get("_bg_writeback_idx")
        bg_w_count = int(bg_w_idx.numel()) if bg_w_idx is not None else num_gaussians_bg
        writeback_bg_ratio = float(bg_w_count / max(num_gaussians_bg, 1))
        distant_w_idx = out.get("_distant_writeback_idx")
        distant_w_count = int(distant_w_idx.numel()) if distant_w_idx is not None else num_gaussians_distant
        writeback_distant_ratio = float(distant_w_count / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0
        hidden_stats = out.get("_hidden_stats", {})
        offset_stats = out.get("_offset_stats", {})
        frame_loss_map = out.get("_frame_loss_map", {})
        num_bg_src_feat_valid = int(out.get("_num_bg_src_feat_valid", 0))
        num_bg_update = int(out.get("_num_bg_update", 0))
        num_distant_src_feat_valid = int(out.get("_num_distant_src_feat_valid", 0))
        num_distant_update = int(out.get("_num_distant_update", 0))

        return {
            "loss": float(loss.detach().item()),
            "loss_l1": out["loss_l1"].item() if torch.is_tensor(out.get("loss_l1")) else float(out.get("loss_l1", 0.0)),
            "loss_ssim": out["loss_ssim"].item() if torch.is_tensor(out.get("loss_ssim")) else float(out.get("loss_ssim", 0.0)),
            "loss_mask": out["loss_mask"].item() if torch.is_tensor(out.get("loss_mask")) else float(out.get("loss_mask", 0.0)),
            "loss_opacity_entropy": out["loss_opacity_entropy"].item()
            if torch.is_tensor(out.get("loss_opacity_entropy"))
            else float(out.get("loss_opacity_entropy", 0.0)),
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": num_gaussians_bg,
            "num_gaussians_distant": num_gaussians_distant,
            "num_gaussians_rigid": num_gaussians_rigid,
            "num_gaussians_sky": 0,
            "num_rigid_valid_src": num_rigid_valid_src,
            "num_rigid_invalid_src": int(max(num_rigid_total - num_rigid_valid_src, 0)),
            "rigid_valid_ratio": float(num_rigid_valid_src / max(num_rigid_total, 1)),
            "num_rigid_src_feat_valid": int(out.get("_num_rigid_src_feat_valid", 0)),
            "num_rigid_update": int(out.get("_num_rigid_update", 0)),
            "rigid_update_ratio": float(out.get("_rigid_update_ratio", 0.0)),
            "rigid_update_among_feat_valid": float(out.get("_rigid_update_among_feat_valid", 0.0)),
            "writeback_rigid_ratio": writeback_rigid_ratio,
            "num_target_frames": int(out.get("_num_target_frames", 0)),
            "loss_effective_frames": int(out.get("_loss_effective_frames", 0)),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "frame_loss_map": frame_loss_map,
            "hidden_norm_bg_mean": float(hidden_stats.get("hidden_norm_bg_mean", 0.0)),
            "hidden_norm_distant_mean": float(hidden_stats.get("hidden_norm_distant_mean", 0.0)),
            "hidden_norm_rigid_mean": float(hidden_stats.get("hidden_norm_rigid_mean", 0.0)),
            "hidden_norm_sky_mean": 0.0,
            "num_sky_src_feat_valid": 0,
            "num_sky_update": 0,
            "sky_update_ratio": 0.0,
            "num_bg_src_feat_valid": num_bg_src_feat_valid,
            "num_bg_update": num_bg_update,
            "bg_update_ratio": float(num_bg_update / max(num_gaussians_bg, 1)),
            "num_distant_src_feat_valid": num_distant_src_feat_valid,
            "num_distant_update": num_distant_update,
            "distant_update_ratio": float(num_distant_update / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0,
            "writeback_bg_ratio": writeback_bg_ratio,
            "writeback_distant_ratio": writeback_distant_ratio,
            "src_backproject_pass_count": int(out.get("_src_backproject_pass_count", 0)),
            "grad_norm_total": float(total_grad_norm),
            "grad_scaler_scale_before": float(scale_before),
            "grad_scaler_scale_after": float(scale_after),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            **{k: float(v) for k, v in offset_stats.items()},
            **grad_norms,
            **timing_ms,
            "grad_norm_sky": 0.0,
            "node_state_sync_update": node_state_sync_update,
            "node_state_sync_reset": node_state_sync_reset,
            "stage5_3_production_fast_fail": 1.0,
        }


__all__ = ["MinimalStreetForwardStage5_3_Production"]
