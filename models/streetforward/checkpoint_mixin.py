from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from models.streetforward.node_states import (
    NodeState,
    NodeStateBackground,
    NodeStateRigid,
    NodeStateDistant,
)
from models.streetforward.stage6_0.local_gs_state import _uncertainty_state_values

logger = logging.getLogger(__name__)


class CheckpointMixin:
    def _appearance_logvar_from_state_dict(
        self,
        state_dict: Dict,
        *,
        branch_name: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        value = state_dict.get("appearance_logvar")
        if torch.is_tensor(value):
            out = value.to(device=self.device, dtype=torch.float32)
            if tuple(out.shape) != (int(ref.shape[0]), 1):
                raise ValueError(
                    f"Checkpoint {branch_name}.appearance_logvar shape mismatch: "
                    f"got {tuple(out.shape)}, expected {(int(ref.shape[0]), 1)}"
                )
            return out
        prior_logvar, _, _, _ = _uncertainty_state_values(
            getattr(self, "iforward_uncertainty_state_cfg", None),
            str(branch_name),
        )
        return torch.full(
            (int(ref.shape[0]), 1),
            float(prior_logvar),
            device=self.device,
            dtype=torch.float32,
        )

    def _node_state_to_dict(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        """
        将 NodeState 转换为字典（用于保存检查点）。
        
        Args:
            node_state: NodeState 对象
            
        Returns:
            状态字典，所有张量都已分离并移到 CPU
        """
        out = {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
        }
        appearance = getattr(node_state, "appearance_logvar", None)
        if torch.is_tensor(appearance):
            out["appearance_logvar"] = appearance.detach().float().cpu()
        return out

    def _node_state_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> NodeState:
        """
        从字典恢复 NodeState（用于加载检查点）。
        
        Args:
            state_dict: 状态字典
            
        Returns:
            恢复的 NodeState，所有张量都已移到设备并分离
        """
        means = state_dict["means"].to(self.device)
        return NodeState(
            means=means,
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
            appearance_logvar=self._appearance_logvar_from_state_dict(
                state_dict,
                branch_name="bg",
                ref=means,
            ),
        ).detach_clone()

    def _node_state_distant_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> NodeStateDistant:
        """
        从字典恢复 NodeStateDistant（用于加载检查点）。
        """
        means = state_dict["means"].to(self.device)
        return NodeStateDistant(
            means=means,
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
            appearance_logvar=self._appearance_logvar_from_state_dict(
                state_dict,
                branch_name="distant",
                ref=means,
            ),
        ).detach_clone()

    def _node_state_rigid_to_dict(self, node_state: NodeStateRigid) -> Dict:
        """
        将 NodeStateRigid 转换为字典（用于保存检查点）。
        
        Args:
            node_state: NodeStateRigid 对象
            
        Returns:
            状态字典，所有张量都已分离并移到 CPU
        """
        out = {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
            "point_ids": node_state.point_ids.detach().cpu(),
            "instances_quats": node_state.instances_quats.detach().cpu(),
            "instances_trans": node_state.instances_trans.detach().cpu(),
            "instances_fv": node_state.instances_fv.detach().cpu(),
            "instance_ids": list(node_state.instance_ids),
            "frame_ids": list(node_state.frame_ids),
            "cur_frame": int(node_state.cur_frame),
        }
        appearance = getattr(node_state, "appearance_logvar", None)
        if torch.is_tensor(appearance):
            out["appearance_logvar"] = appearance.detach().float().cpu()
        return out

    def _node_state_rigid_from_dict(self, state_dict: Dict) -> NodeStateRigid:
        """
        从字典恢复 NodeStateRigid（用于加载检查点）。
        
        Args:
            state_dict: 状态字典
            
        Returns:
            恢复的 NodeStateRigid，所有张量都已移到设备并分离
        """
        instance_ids = state_dict.get("instance_ids")
        if instance_ids is None:
            num_instances = state_dict["instances_quats"].shape[1]
            instance_ids = list(range(num_instances))
        elif isinstance(instance_ids, torch.Tensor):
            instance_ids = instance_ids.tolist()
        means = state_dict["means"].to(self.device)
        return NodeStateRigid(
            means=means,
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
            point_ids=state_dict["point_ids"].to(self.device),
            instances_quats=state_dict["instances_quats"].to(self.device),
            instances_trans=state_dict["instances_trans"].to(self.device),
            instances_fv=state_dict["instances_fv"].to(self.device),
            instance_ids=list(instance_ids),
            frame_ids=list(state_dict.get("frame_ids", [])),
            cur_frame=int(state_dict.get("cur_frame", 0)),
            appearance_logvar=self._appearance_logvar_from_state_dict(
                state_dict,
                branch_name="rigid",
                ref=means,
            ),
        ).detach_clone()

    def save_checkpoint(
        self,
        step: Optional[int] = None,
        is_final: bool = False,
        checkpoint_dir: Optional[str] = None,
    ) -> str:
        """
        持久化模型/优化器和分离的节点状态。
        
        Args:
            step: 可选的训练步数（默认为 self.global_step）
            is_final: 如果为 True，总是写入 checkpoint_final.pth
            checkpoint_dir: 覆盖输出目录
            
        Returns:
            检查点文件路径
        
        保存内容：
        - 模型状态（sparse_conv、所有 MLP 头）
        - 优化器状态
        - 所有 NodeStateBackground（静态背景）
        - 所有 NodeStateRigid（动态物体）
        - 配置（如果可序列化）
        """
        step_val = int(step if step is not None else self.global_step)
        ckpt_dir = (
            checkpoint_dir
            or self.checkpoint_dir
            or (os.path.join(self.config.log_dir, "checkpoints") if hasattr(self.config, "log_dir") else None)
            or "./checkpoints"
        )
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        filename = "checkpoint_final.pth" if is_final else f"checkpoint_step_{step_val:06d}.pth"
        checkpoint_path = os.path.join(ckpt_dir, filename)

        model_state_dict = {
            "sparse_conv": self.sparse_conv.state_dict(),
            "mlp_offset_pos": self.mlp_offset_pos.state_dict(),
            "mlp_conv": self.mlp_conv.state_dict(),
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "gaussion_decoder": self.gaussion_decoder.state_dict(),
        }
        if self.image_feature_extractor is not None:
            model_state_dict["image_feature_extractor"] = self.image_feature_extractor.state_dict()
        # GRU-style modules (may be absent in older checkpoints)
        if hasattr(self, "mlp_params_embed"):
            model_state_dict["mlp_params_embed"] = self.mlp_params_embed.state_dict()
        if hasattr(self, "param_embed_norm"):
            model_state_dict["param_embed_norm"] = self.param_embed_norm.state_dict()
        if hasattr(self, "gru_update"):
            model_state_dict["gru_update"] = self.gru_update.state_dict()
        if hasattr(self, "gru_candidate"):
            model_state_dict["gru_candidate"] = self.gru_candidate.state_dict()
        if hasattr(self, "gru_reset") and self.gru_reset is not None:
            model_state_dict["gru_reset"] = self.gru_reset.state_dict()
        if hasattr(self, "gru_to_head") and not isinstance(self.gru_to_head, torch.nn.Identity):
            model_state_dict["gru_to_head"] = self.gru_to_head.state_dict()

        nodes_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states.items()
        }
        rigid_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_rigid_to_dict(state)
            for (scene, segment), state in self.node_states_rigid.items()
            if state is not None
        }
        distant_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states_distant.items()
            if state is not None
        }

        h_cache_bg = {
            f"scene_{scene}_segment_{segment}": h.detach().cpu()
            for (scene, segment), h in getattr(self, "h_cache_bg", {}).items()
        }
        h_cache_rigid = {
            f"scene_{scene}_segment_{segment}": h.detach().cpu()
            for (scene, segment), h in getattr(self, "h_cache_rigid", {}).items()
        }
        h_cache_distant = {
            f"scene_{scene}_segment_{segment}": h.detach().cpu()
            for (scene, segment), h in getattr(self, "h_cache_distant", {}).items()
        }

        checkpoint = {
            "step": step_val,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "node_states": nodes_state_dict,
        }
        if hasattr(self, "scheduler") and self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if hasattr(self, "grad_scaler") and self.grad_scaler is not None:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        checkpoint.update({
            "node_states_rigid": rigid_state_dict,
            "node_states_distant": distant_state_dict,
            "h_cache_bg": h_cache_bg,
            "h_cache_rigid": h_cache_rigid,
            "h_cache_distant": h_cache_distant,
        })
        try:
            checkpoint["config"] = OmegaConf.to_container(self.config, resolve=False)
        except Exception:
            logger.debug("Config not serialized into checkpoint (non-fatal).")

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> int:
        """
        恢复模型/优化器和节点状态。
        
        Args:
            checkpoint_path: .pth 检查点文件路径
            load_optimizer: 如果可用，加载优化器状态
            strict: 权重加载的严格性
            
        Returns:
            恢复的 global_step
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        self.sparse_conv.load_state_dict(model_state["sparse_conv"], strict=strict)
        self.mlp_offset_pos.load_state_dict(model_state["mlp_offset_pos"], strict=strict)
        self.mlp_conv.load_state_dict(model_state["mlp_conv"], strict=strict)
        self.mlp_opacity.load_state_dict(model_state["mlp_opacity"], strict=strict)
        self.gaussion_decoder.load_state_dict(model_state["gaussion_decoder"], strict=strict)
        if "image_feature_extractor" in model_state and self.image_feature_extractor is not None:
            self.image_feature_extractor.load_state_dict(model_state["image_feature_extractor"], strict=strict)
        # Optional GRU-style modules (backward compatible)
        if "mlp_params_embed" in model_state and hasattr(self, "mlp_params_embed"):
            self.mlp_params_embed.load_state_dict(model_state["mlp_params_embed"], strict=False)
        if "param_embed_norm" in model_state and hasattr(self, "param_embed_norm"):
            self.param_embed_norm.load_state_dict(model_state["param_embed_norm"], strict=False)
        if "gru_update" in model_state and hasattr(self, "gru_update"):
            self.gru_update.load_state_dict(model_state["gru_update"], strict=False)
        if "gru_candidate" in model_state and hasattr(self, "gru_candidate"):
            self.gru_candidate.load_state_dict(model_state["gru_candidate"], strict=False)
        if "gru_reset" in model_state and hasattr(self, "gru_reset") and self.gru_reset is not None:
            self.gru_reset.load_state_dict(model_state["gru_reset"], strict=False)
        if "gru_to_head" in model_state and hasattr(self, "gru_to_head") and not isinstance(self.gru_to_head, torch.nn.Identity):
            self.gru_to_head.load_state_dict(model_state["gru_to_head"], strict=False)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler") and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.debug("Scheduler state restored from checkpoint")
            except Exception as e:
                logger.warning(f"Scheduler state restore failed (non-fatal): {e}")
        if load_optimizer and "grad_scaler_state_dict" in checkpoint and hasattr(self, "grad_scaler") and self.grad_scaler is not None:
            try:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
                logger.debug("GradScaler state restored from checkpoint")
            except Exception as e:
                logger.warning(f"GradScaler state restore failed (non-fatal): {e}")

        nodes_state_dict = checkpoint.get("node_states") or checkpoint.get("nodes_state_dict")
        if nodes_state_dict is not None:
            restored_nodes: Dict[Tuple[int, int], NodeState] = {}
            for key, state in nodes_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_nodes[(scene_id, segment_id)] = self._node_state_from_dict(state)
            if restored_nodes:
                self.node_states = restored_nodes
                self.node_states_bg = self.node_states

        rigid_state_dict = checkpoint.get("node_states_rigid")
        if rigid_state_dict is not None:
            restored_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
            for key, state in rigid_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_rigid[(scene_id, segment_id)] = self._node_state_rigid_from_dict(state)
            if restored_rigid:
                self.node_states_rigid = restored_rigid

        distant_state_dict = checkpoint.get("node_states_distant")
        if distant_state_dict is not None:
            restored_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]] = {}
            for key, state in distant_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_distant[(scene_id, segment_id)] = self._node_state_distant_from_dict(state)
            if restored_distant:
                self.node_states_distant = restored_distant

        # Restore hidden caches if present
        self.h_cache_bg = {}
        self.h_cache_rigid = {}
        self.h_cache_distant = {}
        for cache_key, cache_attr in [
            ("h_cache_bg", "h_cache_bg"),
            ("h_cache_rigid", "h_cache_rigid"),
            ("h_cache_distant", "h_cache_distant"),
        ]:
            cache_dict = checkpoint.get(cache_key, {})
            restored: Dict[Tuple[int, int], torch.Tensor] = {}
            for key, tensor in cache_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored[(scene_id, segment_id)] = tensor.to(self.device)
            setattr(self, cache_attr, restored)

        self.global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
        logger.info(f"Checkpoint loaded from {checkpoint_path} (step={self.global_step})")
        return self.global_step
