"""
Training script for StreetForward feed-forward 3DGS.

This script implements the training loop for StreetForwardTrainer,
supporting multi-scene, multi-segment training with RGB point cloud initialization.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from models.trainers.streetforward import StreetForwardTrainer
from utils.logging import MetricLogger, setup_logging
from utils.streetforward_baseline import set_deterministic_seed

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def set_seeds(seed=31):
    """Fix random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup(args):
    """Setup configuration and logging."""
    global logger
    # Load base config
    cfg = OmegaConf.load(args.config_file)
    
    # Parse CLI arguments
    args_from_cli = OmegaConf.from_cli(args.opts)
    
    # Handle dataset preset/type from CLI (data.dataset_preset or data.dataset)
    dataset_type = None
    dataset_preset = None
    if "data" in args_from_cli:
        if "dataset_preset" in args_from_cli.data:
            dataset_preset = args_from_cli.data.dataset_preset
            del args_from_cli.data.dataset_preset
        if "dataset" in args_from_cli.data:
            dataset_type = args_from_cli.data.dataset
            del args_from_cli.data.dataset
    if dataset_preset is None and "dataset_preset" in args_from_cli:
        dataset_preset = args_from_cli.pop("dataset_preset")
    if dataset_type is None and "dataset" in args_from_cli:
        # If dataset is passed as top-level, it should be the dataset type
        dataset_type = args_from_cli.pop("dataset")
    
    # Get dataset preset/type from config if not in CLI
    if dataset_preset is None and "data" in cfg and hasattr(cfg.data, "get"):
        dataset_preset = cfg.data.get("dataset_preset")
    if dataset_type is None and "data" in cfg and hasattr(cfg.data, "dataset"):
        dataset_type = cfg.data.dataset
    
    dataset_cfg_path = None
    dataset_cfg_key = dataset_preset or dataset_type
    if dataset_cfg_key is not None:
        candidate_path = os.path.join("configs", "datasets", f"{dataset_cfg_key}.yaml")
        if os.path.exists(candidate_path):
            dataset_cfg_path = candidate_path
        else:
            logger.warning(
                f"Dataset preset '{dataset_cfg_key}' not found at {candidate_path}; skipping preset merge."
            )
    
    # Load dataset config if available
    if dataset_cfg_path is not None:
        dataset_cfg = OmegaConf.load(dataset_cfg_path)
        cfg = OmegaConf.merge(cfg, dataset_cfg)
        # Persist the preset key for traceability
        if "data" in cfg:
            cfg.data["dataset_preset"] = dataset_cfg_key
    elif dataset_preset is not None and "data" in cfg:
        # Preserve requested preset even if the preset file is missing
        cfg.data["dataset_preset"] = dataset_preset
    
    # Merge CLI arguments
    cfg = OmegaConf.merge(cfg, args_from_cli)
    
    # Tiny overfit: 1 scene, default 500 steps (override with training.max_iterations=N in opts)
    if getattr(args, "tiny_overfit", False):
        cfg.data.train_scene_ids = [0]
        if not any("max_iterations" in str(o) for o in (getattr(args, "opts", None) or [])):
            cfg.training.max_iterations = 500
    if "data" not in cfg:
        raise ValueError("data config is required but not found")
    
    if "dataset" not in cfg:
        raise ValueError("dataset config is required but not found")
    
    if "model" not in cfg:
        raise ValueError("model config is required but not found")
    
    # Create log directory
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories
    for folder in ["images", "videos", "metrics", "configs_bk", "checkpoints"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # Setup logging
    log_level_name = None
    if "training" in cfg and hasattr(cfg.training, "get"):
        log_level_name = cfg.training.get("log_level", None)
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO) if log_level_name is not None else logging.INFO

    setup_logging(output=log_dir, level=log_level, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Config saved to {saved_cfg_path}")
    
    return cfg


def convert_batch_to_streetforward_format(
    batch: Dict,
    device: torch.device,
) -> Dict:
    """
    将 MultiSceneDataset 的 batch 格式转换为 StreetForwardTrainer 期望的格式。
    
    Args:
        batch: MultiSceneDataset 返回的 batch
        device: 设备
        
    Returns:
        StreetForwardTrainer 期望的 batch 格式
    """
    # 提取基本信息
    scene_id = batch["scene_id"]
    segment_id = batch["segment_id"]
    if isinstance(segment_id, int):
        segment_id = torch.tensor([segment_id], dtype=torch.long)
    
    # 提取点云数据
    pointcloud = batch.get("pointcloud")
    if pointcloud is None:
        raise ValueError("pointcloud is required but not found in batch")
    dynamic_info = batch.get("dynamic_info")
    
    # 转换 target 视图
    target_data = batch["target"]
    target_views = []
    gt_images = []
    targets = []
    
    num_target_images = target_data["image"].shape[0]  # [N, H, W, 3]
    for i in range(num_target_images):
        # 创建 View 对象
        view = type('View', (), {
            'camtoworlds': target_data["extrinsics"][i].to(device),  # [4, 4]
            'Ks': target_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),  # [1, 3, 3]
        })()
        target_views.append(view)
        
        # 提取 GT 图像
        gt_image = target_data["image"][i].to(device)  # [H, W, 3]
        gt_images.append(gt_image)
        frame_indices = target_data.get("frame_indices")
        frame_idx = int(frame_indices[i]) if frame_indices is not None else 0
        targets.append({
            "frame_idx": frame_idx,
            "view": view,
            "gt_image": gt_image,
        })
    
    # 转换 source 视图（可选，用于未来扩展）
    source_views = []
    src_images = []
    source_frame_idx = None
    if "source" in batch:
        source_data = batch["source"]
        num_source_images = source_data["image"].shape[0]
        for i in range(num_source_images):
            view = type('View', (), {
                'camtoworlds': source_data["extrinsics"][i].to(device),
                'Ks': source_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
            })()
            source_views.append(view)
            src_image = source_data["image"][i].to(device)
            src_images.append(src_image)
            if source_frame_idx is None:
                frame_indices = source_data.get("frame_indices")
                if frame_indices is not None:
                    source_frame_idx = int(frame_indices[i])

    # 转换 test 视图（可选，用于评估）
    test_views = []
    test_images = []
    if "test" in batch:
        test_data = batch["test"]
        num_test_images = test_data["image"].shape[0]
        for i in range(num_test_images):
            view = type('View', (), {
                'camtoworlds': test_data["extrinsics"][i].to(device),
                'Ks': test_data["intrinsics"][i][:3, :3].unsqueeze(0).to(device),
            })()
            test_views.append(view)
            test_image = test_data["image"][i].to(device)
            test_images.append(test_image)
    
    # 组装 StreetForward 格式的 batch
    streetforward_batch = {
        "scene_id": scene_id.to(device) if isinstance(scene_id, torch.Tensor) else torch.tensor([scene_id], dtype=torch.long).to(device),
        "segment_id": segment_id.to(device) if isinstance(segment_id, torch.Tensor) else torch.tensor([segment_id], dtype=torch.long).to(device),
        "pointcloud": pointcloud,
        "dynamic_info": dynamic_info,
        "target_views": target_views,
        "gt_images": gt_images,
        "targets": targets,
        "source_frame_idx": source_frame_idx if source_frame_idx is not None else 0,
        "source_views": source_views,  # 可选
        "src_images": src_images,  # 可选
        "test_views": test_views,  # 可选
        "test_images": test_images,  # 可选
    }
    
    return streetforward_batch


def main(args):
    """Main training function."""
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Deterministic reproducibility: set seed before any random sampling
    seed = cfg.training.get("seed", 42)
    set_deterministic_seed(seed)
    logger.info(f"Deterministic seed set to {seed} (cudnn.deterministic=True, benchmark=False)")

    # Build MultiSceneDataset
    logger.info("Building MultiSceneDataset...")
    dataset = MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=cfg.data.train_scene_ids,
        eval_scene_ids=cfg.data.eval_scene_ids,
        num_source_keyframes=cfg.dataset.num_source_keyframes,
        num_target_keyframes=cfg.dataset.num_target_keyframes,
        segment_overlap_ratio=cfg.dataset.segment_overlap_ratio,
        keyframe_split_config=cfg.dataset.get("keyframe_split_config", None),
        min_keyframes_per_scene=cfg.dataset.min_keyframes_per_scene,
        min_keyframes_per_segment=cfg.dataset.min_keyframes_per_segment,
        device=device,
        preload_scene_count=cfg.dataset.get("preload_scene_count", 3),
        fixed_segment_aabb=cfg.dataset.get("fixed_segment_aabb", None),
        pointcloud_config=cfg.dataset.get("pointcloud", None),  # 传递点云配置
    )
    logger.info(f"Dataset initialized with {len(dataset.train_scene_ids)} training scenes")
    
    # Tiny overfit: use fixed (scene_id=0, segment_id=0) batch every step
    fixed_batch = None
    if getattr(args, "tiny_overfit", False):
        dataset.initialize()
        fixed_batch = dataset.get_segment_batch(0, 0, include_test=True)
        logger.info("Tiny overfit: using fixed batch scene_id=0, segment_id=0")
    
    # Build trainer
    logger.info("Building StreetForwardTrainer...")
    trainer = StreetForwardTrainer(
        config=cfg,
        device=device,
    )

    # Optionally resume from checkpoint
    resume_path = args.resume or cfg.training.get("resume_from_checkpoint", None)
    if resume_path:
        try:
            restored_step = trainer.load_checkpoint(resume_path, load_optimizer=True)
            logger.info(f"Resumed from checkpoint at step {restored_step}")
        except Exception as exc:
            logger.error(f"Failed to resume from {resume_path}: {exc}", exc_info=True)
            raise

    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    
    # Training loop
    max_iterations = cfg.training.max_iterations
    save_checkpoint_freq = cfg.training.save_checkpoint_freq
    log_interval = cfg.training.get("log_interval", 100)
    eval_freq = cfg.training.get("eval_freq", 5000)

    logger.info(f"Starting training for {max_iterations} iterations")
    logger.info(
        f"Training setup: use_amp={getattr(trainer, 'use_amp', False)}, "
        f"grad_clip_max_norm={getattr(trainer, 'grad_clip_max_norm', None)}, "
        f"scheduler={type(trainer.scheduler).__name__ if getattr(trainer, 'scheduler', None) else None}"
    )

    trainer.train()

    try:
        step = getattr(trainer, "global_step", 0)
        while step < max_iterations:
            # Sample random batch or use fixed batch (tiny overfit)
            if fixed_batch is not None:
                batch = fixed_batch
            else:
                batch = dataset.sample_random_batch()
            
            # Convert batch format
            streetforward_batch = convert_batch_to_streetforward_format(batch, device)
            
            # Training step
            try:
                result = trainer.train_iter(streetforward_batch, apply_update=True, update_state=True)
            except RuntimeError as e:
                logger.error(
                    f"Step {step} failed (RuntimeError): {e}. "
                    "Check: OOM, AMP compatibility, or invalid tensors."
                )
                raise
            except ValueError as e:
                logger.error(
                    f"Step {step} failed (ValueError): {e}. "
                    "Likely causes: empty targets or invalid/empty pointcloud inputs."
                )
                raise
            step = getattr(trainer, "global_step", step + 1)
            current_lr = None
            grad_norm = getattr(trainer, "_last_grad_norm", None)
            if hasattr(trainer, "optimizer"):
                try:
                    current_lr = trainer.optimizer.param_groups[0]["lr"]
                except Exception:
                    current_lr = None
            
            # Update metrics
            total_loss = result.get("total_loss", torch.tensor(0.0, device=device))
            loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            if not (float("-inf") < loss_val < float("inf")):
                logger.warning(f"Step {step}: loss is NaN or inf ({loss_val}). Check data and model.")
            metric_logger.update(loss=loss_val)
            metric_logger.update(step=step)
            if current_lr is not None:
                metric_logger.update(lr=current_lr)
            if grad_norm is not None:
                metric_logger.update(grad_norm=grad_norm)

            # Logging (trainer._log_to_tensorboard already writes lr/grad_norm per tb_log_every)
            if step % log_interval == 0:
                logger.info(f"Step {step}: {metric_logger}")

            if step > 0 and (step % save_checkpoint_freq == 0 or step == max_iterations - 1):
                trainer.save_checkpoint(step=step, is_final=(step >= max_iterations - 1))

            # Evaluation (可选；tiny overfit 时跳过以加快运行)
            if step > 0 and step % eval_freq == 0 and fixed_batch is None:
                logger.info("Running evaluation...")
                trainer.eval()
                
                eval_metrics = []
                num_eval_batches = min(10, len(dataset.eval_scene_ids) * 3)
                
                for _ in range(num_eval_batches):
                    try:
                        eval_batch = dataset.sample_random_batch(eval=True, include_test=True)
                        eval_streetforward_batch = convert_batch_to_streetforward_format(eval_batch, device)
                        
                        # 评估（包含测试视角指标）
                        with torch.no_grad():
                            result = trainer.train_iter(
                                eval_streetforward_batch,
                                apply_update=False,
                                update_state=False,
                                evaluate_test=True,
                            )
                            metric_entry = {"loss": result.get("total_loss", torch.tensor(0.0)).item()}
                            if result.get("test_metrics") is not None:
                                metric_entry.update(result["test_metrics"])
                            eval_metrics.append(metric_entry)
                    except Exception as e:
                        logger.warning(f"Evaluation batch failed: {e}")
                        continue
                
                if eval_metrics:
                    avg_loss = np.mean([m["loss"] for m in eval_metrics])
                    logger.info(f"Evaluation loss: {avg_loss}")
                    metric_logger.update(eval_loss=avg_loss)
                    
                    psnr_vals = [m["psnr"] for m in eval_metrics if "psnr" in m]
                    ssim_vals = [m["ssim"] for m in eval_metrics if "ssim" in m]
                    lpips_vals = [m["lpips"] for m in eval_metrics if "lpips" in m]
                    if psnr_vals:
                        avg_psnr = np.mean(psnr_vals)
                        logger.info(
                            f"Evaluation PSNR: {avg_psnr:.4f}, "
                            f"SSIM: {np.mean(ssim_vals) if ssim_vals else float('nan'):.4f}, "
                            f"LPIPS: {np.mean(lpips_vals) if lpips_vals else float('nan'):.4f}"
                        )
                        if hasattr(trainer, "step_scheduler_plateau"):
                            trainer.step_scheduler_plateau(avg_psnr)
                    elif hasattr(trainer, "step_scheduler_plateau"):
                        trainer.step_scheduler_plateau(-avg_loss)
                
                trainer.train()
        
        logger.info("Training completed!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if hasattr(trainer, "close"):
            trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StreetForward feed-forward 3DGS")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./logs",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="streetforward",
        help="Project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (default: timestamp)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--tiny_overfit",
        action="store_true",
        help="Tiny overfit: 1 scene (0), 1 segment (0), fixed batch, 500 steps (override with training.max_iterations=N).",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional configuration options",
    )
    
    args = parser.parse_args()
    
    # Set default run name
    if args.run_name is None:
        args.run_name = current_time
    
    main(args)
