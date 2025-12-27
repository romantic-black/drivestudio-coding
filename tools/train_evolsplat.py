"""
Training script for EVolsplat feed-forward 3DGS.

This script implements the training loop for EVolsplatTrainer,
supporting multi-scene, multi-segment training with RGB point cloud initialization.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from datasets.pointcloud_generators.rgb_pointcloud_generator import MonocularRGBPointCloudGenerator
from models.trainers.evolsplat import EVolsplatTrainer
from utils.logging import MetricLogger, setup_logging

logger = logging.getLogger(__name__)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def set_seeds(seed=31):
    """Fix random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup(args):
    """Setup configuration and logging."""
    # Load base config
    cfg = OmegaConf.load(args.config_file)
    
    # Parse CLI arguments
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
    
    # Load dataset config if specified (similar to train.py)
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # Merge dataset config
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # Merge CLI arguments
    cfg = OmegaConf.merge(cfg, args_from_cli)
    
    # Ensure required config keys exist
    # If data is not in cfg, try to load from multi_scene.yaml
    if "data" not in cfg:
        multi_scene_cfg_path = os.path.join("configs", "evolsplat", "multi_scene.yaml")
        if os.path.exists(multi_scene_cfg_path):
            multi_scene_cfg = OmegaConf.load(multi_scene_cfg_path)
            if "data" in multi_scene_cfg:
                cfg.data = multi_scene_cfg.data
            if "multi_scene" in multi_scene_cfg:
                cfg.multi_scene = multi_scene_cfg.multi_scene
    
    # If trainer is not in cfg, try to load from trainer_config.yaml
    if "trainer" not in cfg:
        trainer_cfg_path = os.path.join("configs", "evolsplat", "trainer_config.yaml")
        if os.path.exists(trainer_cfg_path):
            trainer_cfg = OmegaConf.load(trainer_cfg_path)
            if "trainer" in trainer_cfg:
                cfg.trainer = trainer_cfg.trainer
    
    # Ensure data config exists
    if "data" not in cfg:
        raise ValueError("data config is required but not found")
    
    # Add default pointcloud config if missing (under data.pointcloud)
    if "pointcloud" not in cfg.data:
        raise ValueError("pointcloud config is required but not found")
    
    # Create log directory
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories
    for folder in ["images", "videos", "metrics", "configs_bk", "checkpoints"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # Setup logging
    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Config saved to {saved_cfg_path}")
    
    return cfg


def main(args):
    """Main training function."""
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build MultiSceneDataset
    logger.info("Building MultiSceneDataset...")
    dataset = MultiSceneDataset(
        data_cfg=cfg.data,
        train_scene_ids=cfg.data.train_scene_ids,
        eval_scene_ids=cfg.data.eval_scene_ids,
        num_source_keyframes=cfg.multi_scene.num_source_keyframes,
        num_target_keyframes=cfg.multi_scene.num_target_keyframes,
        segment_overlap_ratio=cfg.multi_scene.segment_overlap_ratio,
        keyframe_split_config=cfg.multi_scene.get("keyframe_split_config", None),
        min_keyframes_per_scene=cfg.multi_scene.min_keyframes_per_scene,
        min_keyframes_per_segment=cfg.multi_scene.min_keyframes_per_segment,
        device=device,
        preload_scene_count=cfg.multi_scene.get("preload_scene_count", 3),
        fixed_segment_aabb=cfg.multi_scene.get("fixed_segment_aabb", None),
    )
    logger.info(f"Dataset initialized with {len(dataset.train_scene_ids)} training scenes")
    
    # Build RGB point cloud generator
    logger.info("Building RGBPointCloudGenerator...")
    crop_aabb = np.array(cfg.data.pointcloud.crop_aabb)
    input_aabb = np.array(cfg.data.pointcloud.input_aabb)
    pointcloud_generator = MonocularRGBPointCloudGenerator(
        chosen_cam_ids=cfg.data.pixel_source.cameras,
        sparsity=cfg.data.pointcloud.get("sparsity", "full"),
        filter_sky=cfg.data.pointcloud.get("filter_sky", True),
        depth_consistency=cfg.data.pointcloud.get("depth_consistency", True),
        use_bbx=cfg.data.pointcloud.get("use_bbx", True),
        downscale=cfg.data.pointcloud.get("downscale", 2),
        crop_aabb=crop_aabb,
        input_aabb=input_aabb,
        device=device,
    )
    
    # Build trainer
    logger.info("Building EVolsplatTrainer...")
    trainer = EVolsplatTrainer(
        dataset=dataset,
        pointcloud_generator=pointcloud_generator,
        config=cfg.trainer,
        device=device,
        log_dir=cfg.log_dir,
    )
    
    # Load checkpoint if resuming
    if args.resume_from is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        step = trainer.load_checkpoint(args.resume_from, load_only_model=False)
        logger.info(f"Resumed to step {step}")
    else:
        step = 0
        logger.info("Starting training from scratch")
    
    # Initialize metric logger
    metric_logger = MetricLogger(delimiter="  ")
    
    # Training loop
    max_iterations = cfg.trainer.training.max_iterations
    save_checkpoint_freq = cfg.trainer.training.save_checkpoint_freq
    
    logger.info(f"Starting training for {max_iterations} iterations")
    
    trainer.set_train()
    
    try:
        while step < max_iterations:
            # Sample random batch
            batch = dataset.sample_random_batch()
            
            # Move batch to device
            for key in ["source", "target"]:
                if key in batch:
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)
            
            # Training step
            loss_dict = trainer.train_step(batch)
            
            # Update metrics
            metric_logger.update(**{f"losses/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()})
            metric_logger.update(step=step)
            
            # Logging
            if step % 100 == 0:
                logger.info(f"Step {step}: {metric_logger}")
            
            # Save checkpoint
            if step > 0 and (step % save_checkpoint_freq == 0 or step == max_iterations - 1):
                trainer.save_checkpoint(step, is_final=(step == max_iterations - 1))
            
            # Evaluation (optional)
            if step > 0 and step % cfg.trainer.training.get("eval_freq", 5000) == 0:
                logger.info("Running evaluation...")
                trainer.set_eval()
                
                eval_metrics = []
                num_eval_batches = min(10, len(dataset.eval_scene_ids) * 3)  # Sample a few batches
                
                for _ in range(num_eval_batches):
                    try:
                        eval_batch = dataset.sample_random_batch(eval=True)
                        # Move to device
                        for key in ["source", "target"]:
                            if key in eval_batch:
                                for subkey in eval_batch[key]:
                                    if isinstance(eval_batch[key][subkey], torch.Tensor):
                                        eval_batch[key][subkey] = eval_batch[key][subkey].to(device)
                        
                        metrics = trainer.evaluate(eval_batch)
                        eval_metrics.append(metrics)
                    except Exception as e:
                        logger.warning(f"Evaluation batch failed: {e}")
                        continue
                
                if eval_metrics:
                    avg_metrics = {
                        k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0].keys()
                    }
                    logger.info(f"Evaluation metrics: {avg_metrics}")
                    metric_logger.update(**{f"eval/{k}": v for k, v in avg_metrics.items()})
                
                trainer.set_train()
            
            step += 1
        
        logger.info("Training completed!")
        trainer.save_checkpoint(step, is_final=True)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(step, is_final=False)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        trainer.save_checkpoint(step, is_final=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EVolsplat feed-forward 3DGS")
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
        default="evolsplat",
        help="Project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (default: timestamp)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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

