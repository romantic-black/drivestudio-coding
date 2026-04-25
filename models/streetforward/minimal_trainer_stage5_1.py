"""
Minimal StreetForward Stage 5.1:
- keep Stage4.6 routed rigid/no-sky semantics
- keep Stage5.0 bg + rigid.S_in struct scope
- add fixed-shape KNN attention after xCPE
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import torch

from models.streetforward.minimal_trainer_stage4_6 import BgRigidInGRUInputs, MinimalStreetForwardStage4_6, RigidRoute
from models.streetforward.minimal_trainer_stage5_0 import MinimalStreetForwardStage5_0
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders import StructDecoderInput, StreetForwardXCPEKNNDecoder, cat_param_dict


class MinimalStreetForwardStage5_1(MinimalStreetForwardStage5_0):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_stage5_1_config(config)
        # Bypass Stage5_0.__init__ fast-fail on model.stage/type.
        MinimalStreetForwardStage4_6.__init__(self, config, device, **kwargs)
        self._init_stage5_1_struct_decoder(config)
        self._rebuild_optimizer_after_stage5_modules()

    def _validate_stage5_1_config(self, config) -> None:
        self._validate_stage4_6_config(config)

        model_cfg = self._require_key(config, "model", "config")
        if str(self._require_key(model_cfg, "stage", "model")) != "5_1":
            raise ValueError("Stage5_1 requires model.stage='5_1'.")
        branches_cfg = self._require_key(model_cfg, "branches", "model")
        distant_cfg = self._require_key(branches_cfg, "distant", "model.branches")
        distant_init_cfg = self._require_key(distant_cfg, "init", "model.branches.distant")
        distant_scale_init_cfg = self._require_key(distant_init_cfg, "scale_init", "model.branches.distant.init")
        distant_scale_mode = str(
            self._require_key(distant_scale_init_cfg, "mode", "model.branches.distant.init.scale_init")
        ).strip().lower()
        if distant_scale_mode == "knn":
            raise ValueError(
                "Stage5_1 does not support model.branches.distant.init.scale_init.mode='knn'. "
                "Use mode='isotropic'."
            )

        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        if not bool(self._require_key(struct_cfg, "enable", "model.struct_decoder")):
            raise ValueError("Stage5_1 requires model.struct_decoder.enable=true.")
        if str(self._require_key(struct_cfg, "type", "model.struct_decoder")) != "xcpe_knn_attn":
            raise ValueError("Stage5_1 struct_decoder.type must be 'xcpe_knn_attn'.")
        if str(self._require_key(struct_cfg, "scope", "model.struct_decoder")) != "bg_rigid_in":
            raise ValueError("Stage5_1 struct_decoder.scope must be 'bg_rigid_in'.")
        if bool(self._require_key(struct_cfg, "include_distant", "model.struct_decoder")):
            raise ValueError("Stage5_1 struct_decoder.include_distant must be false.")
        if bool(self._require_key(struct_cfg, "include_rigid_out", "model.struct_decoder")):
            raise ValueError("Stage5_1 struct_decoder.include_rigid_out must be false.")
        if not bool(self._require_key(struct_cfg, "point_preserving", "model.struct_decoder")):
            raise ValueError("Stage5_1 struct_decoder.point_preserving must be true.")
        if bool(struct_cfg.get("clamp_grid_coord", False)):
            raise ValueError("Stage5_1 does not support struct_decoder.clamp_grid_coord=true.")
        sparse_backend = str(self._require_key(struct_cfg, "sparse_backend", "model.struct_decoder")).lower()
        if sparse_backend != "spconv":
            raise ValueError("Stage5_1 struct_decoder.sparse_backend must be 'spconv'.")
        output_role = str(self._require_key(struct_cfg, "output_role", "model.struct_decoder"))
        if output_role != "gru_input":
            raise ValueError("Stage5_1 struct_decoder.output_role must be 'gru_input'.")

        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        if bool(token_cfg.get("use_hidden_state", False)):
            raise ValueError("Stage5_1 does not support struct_decoder.token.use_hidden_state=true.")
        if bool(token_cfg.get("use_anchor_rgb", False)):
            raise ValueError("Stage5_1 does not support struct_decoder.token.use_anchor_rgb=true.")

        future_cfg = self._require_key(struct_cfg, "future", "model.struct_decoder")
        if bool(self._require_key(future_cfg, "allow_pooling", "model.struct_decoder.future")):
            raise ValueError("Stage5_1 struct_decoder.future.allow_pooling must be false.")
        if bool(self._require_key(future_cfg, "allow_serialized_attention", "model.struct_decoder.future")):
            raise ValueError("Stage5_1 struct_decoder.future.allow_serialized_attention must be false.")
        if bool(self._require_key(future_cfg, "allow_knn_update", "model.struct_decoder.future")):
            raise ValueError("Stage5_1 struct_decoder.future.allow_knn_update must be false.")

        knn_cfg = self._require_key(struct_cfg, "knn_attention", "model.struct_decoder")
        if not bool(self._require_key(knn_cfg, "enable", "model.struct_decoder.knn_attention")):
            raise ValueError("Stage5_1 requires knn_attention.enable=true.")
        if str(self._require_key(knn_cfg, "neighbor_policy", "model.struct_decoder.knn_attention")) != "fixed_cached":
            raise ValueError("Stage5_1 only supports knn_attention.neighbor_policy='fixed_cached'.")
        if (
            str(self._require_key(knn_cfg, "out_neighbor_policy", "model.struct_decoder.knn_attention"))
            != "mask_self_fallback"
        ):
            raise ValueError("Stage5_1 only supports out_neighbor_policy='mask_self_fallback'.")
        if int(self._require_key(knn_cfg, "k", "model.struct_decoder.knn_attention")) <= 1:
            raise ValueError("Stage5_1 knn_attention.k must be > 1.")
        if int(self._require_key(knn_cfg, "num_layers", "model.struct_decoder.knn_attention")) <= 0:
            raise ValueError("Stage5_1 knn_attention.num_layers must be > 0.")
        if int(self._require_key(knn_cfg, "attn_dim", "model.struct_decoder.knn_attention")) <= 0:
            raise ValueError("Stage5_1 knn_attention.attn_dim must be > 0.")
        if int(self._require_key(knn_cfg, "pos_dim", "model.struct_decoder.knn_attention")) <= 0:
            raise ValueError("Stage5_1 knn_attention.pos_dim must be > 0.")
        if int(self._require_key(knn_cfg, "chunk_size", "model.struct_decoder.knn_attention")) <= 0:
            raise ValueError("Stage5_1 knn_attention.chunk_size must be > 0.")
        if float(self._require_key(knn_cfg, "pos_scale", "model.struct_decoder.knn_attention")) <= 0.0:
            raise ValueError("Stage5_1 knn_attention.pos_scale must be > 0.")
        if int(self._require_key(knn_cfg, "min_valid_neighbors", "model.struct_decoder.knn_attention")) < 1:
            raise ValueError("Stage5_1 knn_attention.min_valid_neighbors must be >= 1.")
        if int(self._require_key(knn_cfg, "min_valid_neighbors", "model.struct_decoder.knn_attention")) > (
            int(self._require_key(knn_cfg, "k", "model.struct_decoder.knn_attention")) - 1
        ):
            raise ValueError("Stage5_1 knn_attention.min_valid_neighbors must be <= k-1.")

        debug_cfg = self._require_key(config, "debug", "config")
        _ = bool(self._require_key(debug_cfg, "validate_stage5_knn", "debug"))

    def _init_stage5_1_struct_decoder(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        struct_cfg = self._require_key(model_cfg, "struct_decoder", "model")
        token_cfg = self._require_key(struct_cfg, "token", "model.struct_decoder")
        xcpe_cfg = self._require_key(struct_cfg, "xcpe", "model.struct_decoder")
        knn_cfg = self._require_key(struct_cfg, "knn_attention", "model.struct_decoder")

        feat_2d_channels_cfg = int(self._require_key(struct_cfg, "feat_2d_channels", "model.struct_decoder"))
        feat_2d_channels_model = int(self._require_key(model_cfg, "feat_2d_channels", "model"))
        if feat_2d_channels_cfg != feat_2d_channels_model:
            raise ValueError(
                "Stage5_1 struct_decoder.feat_2d_channels must match model.feat_2d_channels "
                f"({feat_2d_channels_model}), got {feat_2d_channels_cfg}."
            )

        output_dim_cfg = struct_cfg.get("output_dim", "auto")
        output_dim = self.fused_in_dim if str(output_dim_cfg) == "auto" else int(output_dim_cfg)
        if int(output_dim) != int(self.fused_in_dim):
            raise ValueError(
                "Stage5_1 struct_decoder.output_dim must match GRU input dim self.fused_in_dim "
                f"({self.fused_in_dim}), got {output_dim}."
            )

        self.struct_decoder = StreetForwardXCPEKNNDecoder(
            feat_2d_channels=feat_2d_channels_cfg,
            out_channels=int(output_dim),
            param_dim=17,
            branch_embed_dim=int(self._require_key(struct_cfg, "branch_embed_dim", "model.struct_decoder")),
            support_embed_dim=int(self._require_key(struct_cfg, "support_embed_dim", "model.struct_decoder")),
            param_embed_dim=int(self._require_key(struct_cfg, "param_embed_dim", "model.struct_decoder")),
            channels=int(self._require_key(struct_cfg, "channels", "model.struct_decoder")),
            voxel_size=float(self._require_key(struct_cfg, "voxel_size", "model.struct_decoder")),
            xcpe_num_layers=int(self._require_key(xcpe_cfg, "num_layers", "model.struct_decoder.xcpe")),
            xcpe_kernel_size=int(self._require_key(xcpe_cfg, "kernel_size", "model.struct_decoder.xcpe")),
            xcpe_residual_scale_init=float(self._require_key(xcpe_cfg, "residual_scale_init", "model.struct_decoder.xcpe")),
            sparse_backend=str(self._require_key(struct_cfg, "sparse_backend", "model.struct_decoder")),
            norm=str(self._require_key(xcpe_cfg, "norm", "model.struct_decoder.xcpe")),
            act=str(self._require_key(xcpe_cfg, "act", "model.struct_decoder.xcpe")),
            knn_num_layers=int(self._require_key(knn_cfg, "num_layers", "model.struct_decoder.knn_attention")),
            knn_attn_dim=int(self._require_key(knn_cfg, "attn_dim", "model.struct_decoder.knn_attention")),
            knn_pos_dim=int(self._require_key(knn_cfg, "pos_dim", "model.struct_decoder.knn_attention")),
            knn_pos_scale=float(self._require_key(knn_cfg, "pos_scale", "model.struct_decoder.knn_attention")),
            knn_chunk_size=int(self._require_key(knn_cfg, "chunk_size", "model.struct_decoder.knn_attention")),
            knn_residual_scale_init=float(
                self._require_key(knn_cfg, "residual_scale_init", "model.struct_decoder.knn_attention")
            ),
            knn_use_same_branch_flag=bool(
                self._require_key(knn_cfg, "use_same_branch_flag", "model.struct_decoder.knn_attention")
            ),
            knn_use_support=bool(self._require_key(knn_cfg, "use_support", "model.struct_decoder.knn_attention")),
            knn_use_pos_value=bool(self._require_key(knn_cfg, "use_pos_value", "model.struct_decoder.knn_attention")),
            debug_validate=bool(self._require_key(self._require_key(config, "debug", "config"), "validate_stage5_knn", "debug")),
            use_2d_feat=bool(self._require_key(token_cfg, "use_2d_feat", "model.struct_decoder.token")),
            use_support=bool(self._require_key(token_cfg, "use_support", "model.struct_decoder.token")),
            use_branch_embed=bool(self._require_key(token_cfg, "use_branch_embed", "model.struct_decoder.token")),
            use_param_embed=bool(self._require_key(token_cfg, "use_param_embed", "model.struct_decoder.token")),
            zero_invalid_2d_feat=bool(self._require_key(token_cfg, "zero_invalid_2d_feat", "model.struct_decoder.token")),
            clamp_grid_coord=bool(struct_cfg.get("clamp_grid_coord", False)),
        ).to(self.device)

        self.stage5_struct_enabled = True
        self.stage5_1_knn_cfg = {k: knn_cfg.get(k) for k in knn_cfg.keys()}
        cfg_hash_src = self._to_hashable_struct_cfg(struct_cfg)
        self.struct_decoder_cfg_hash = hashlib.sha1(cfg_hash_src.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_hashable_struct_cfg(struct_cfg: Any) -> str:
        if isinstance(struct_cfg, dict):
            serializable = struct_cfg
        else:
            try:
                serializable = {k: struct_cfg.get(k) for k in struct_cfg.keys()}
            except Exception:
                serializable = str(struct_cfg)
        return json.dumps(serializable, sort_keys=True, default=str)

    def _get_segment_knn_tensors(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if batch is None:
            raise ValueError("Stage5_1 requires current batch for KNN tensor resolution.")
        payload = batch.get("knn_struct_neighbors")
        if not isinstance(payload, dict):
            # Backward compatibility: allow reading neighbors directly from batch["knn_init"].
            # This also helps when training scripts forget to forward knn_struct_neighbors.
            maybe_knn_init = batch.get("knn_init")
            if isinstance(maybe_knn_init, dict):
                payload = maybe_knn_init
            else:
                raise ValueError(
                    "Stage5_1 requires batch['knn_struct_neighbors'] (or batch['knn_init']) "
                    "with bg_knn_idx/rigid_knn_idx. KNN asset pipeline is expected to provide this payload."
                )
        if "bg_knn_idx" not in payload or "rigid_knn_idx" not in payload:
            raise ValueError(
                "KNN neighbor payload must contain bg_knn_idx and rigid_knn_idx "
                "(from batch['knn_struct_neighbors'] or batch['knn_init'])."
            )
        bg_knn_idx = torch.as_tensor(payload["bg_knn_idx"])
        rigid_knn_idx = torch.as_tensor(payload["rigid_knn_idx"])
        if bg_knn_idx.dim() != 2:
            raise ValueError(f"bg_knn_idx must be [N_bg,K_store], got {tuple(bg_knn_idx.shape)}")
        if rigid_knn_idx.dim() != 2:
            raise ValueError(f"rigid_knn_idx must be [N_rigid,K_store], got {tuple(rigid_knn_idx.shape)}")
        knn_cfg = getattr(self, "stage5_1_knn_cfg", {}) or {}
        neighbor_policy = str(knn_cfg.get("neighbor_policy", "")).strip().lower()
        rigid_knn_row_ids = None
        # Stage5_1 fixed_cached semantics require rigid KNN and node_state_rigid to share
        # the same full-segment row-space; mapping metadata is ignored on purpose.
        if "rigid_knn_row_ids" in payload and neighbor_policy != "fixed_cached":
            rigid_knn_row_ids = torch.as_tensor(payload["rigid_knn_row_ids"])
            if rigid_knn_row_ids.dim() != 1:
                raise ValueError(f"rigid_knn_row_ids must be [M], got {tuple(rigid_knn_row_ids.shape)}")
            if int(rigid_knn_row_ids.shape[0]) != int(rigid_knn_idx.shape[0]):
                raise ValueError(
                    "rigid_knn_row_ids length must match rigid_knn_idx rows, "
                    f"got {rigid_knn_row_ids.shape[0]} vs {rigid_knn_idx.shape[0]}"
                )
        return bg_knn_idx, rigid_knn_idx, rigid_knn_row_ids

    def _resolve_bg_struct_query_rows(
        self,
        *,
        batch: Dict[str, Any],
        bg_knn_rows: int,
        num_bg: int,
        device: torch.device,
    ) -> torch.Tensor:
        if int(bg_knn_rows) < int(num_bg):
            raise RuntimeError(
                f"bg_knn_idx N mismatch: got {bg_knn_rows}, expected at least {num_bg}"
            )

        pointcloud = batch.get("pointcloud")
        if not isinstance(pointcloud, dict):
            raise RuntimeError(
                "Stage5_1 strict mode requires batch['pointcloud'] dict for bg row-space alignment."
            )
        bg_raw = pointcloud.get("background")
        if bg_raw is None:
            raise RuntimeError(
                "Stage5_1 strict mode requires batch['pointcloud']['background'] for bg row-space alignment."
            )
        bg_all = torch.as_tensor(bg_raw)
        if bg_all.dim() != 2 or int(bg_all.shape[1]) < 3:
            raise RuntimeError(
                "batch['pointcloud']['background'] must have shape [N,>=3] in Stage5_1 strict mode, "
                f"got {tuple(bg_all.shape)}"
            )
        if int(bg_all.shape[0]) != int(bg_knn_rows):
            raise RuntimeError(
                "Stage5_1 strict mode requires bg_knn_idx rows to match pointcloud.background rows: "
                f"knn_rows={bg_knn_rows} pointcloud_rows={int(bg_all.shape[0])}"
            )
        pts = bg_all[:, :3].to(dtype=torch.float32, device="cpu")
        bbx_min = self.bbx_min.detach().to(dtype=torch.float32, device="cpu")
        bbx_max = self.bbx_max.detach().to(dtype=torch.float32, device="cpu")
        in_crop = ((pts >= bbx_min[None, :]) & (pts <= bbx_max[None, :])).all(dim=1)
        near_rows = torch.nonzero(in_crop, as_tuple=False).squeeze(1)
        if int(near_rows.numel()) != int(num_bg):
            raise RuntimeError(
                "Stage5_1 strict mode requires bg query rows to be fully recoverable from pointcloud/background crop. "
                f"near_rows={int(near_rows.numel())} num_bg={int(num_bg)}"
            )
        return near_rows.to(device=device, dtype=torch.long, non_blocking=True)

    def _build_bg_struct_neighbors(
        self,
        *,
        batch: Dict[str, Any],
        bg_knn_idx: torch.Tensor,
        num_bg: int,
        k: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bg_rows = int(bg_knn_idx.shape[0])
        if bg_rows < int(num_bg):
            raise RuntimeError(f"bg_knn_idx N mismatch: got {bg_rows}, expected at least {num_bg}")
        if int(k) <= 1:
            raise RuntimeError(f"Stage5_1 K must be > 1, got {k}")
        k_store = int(k) - 1
        if int(bg_knn_idx.shape[1]) != k_store:
            raise RuntimeError(f"bg_knn_idx K_store mismatch: got {bg_knn_idx.shape[1]}, required {k_store}")

        query_rows = self._resolve_bg_struct_query_rows(
            batch=batch,
            bg_knn_rows=bg_rows,
            num_bg=int(num_bg),
            device=bg_knn_idx.device,
        )
        raw_long_full = bg_knn_idx[query_rows].to(device=device, non_blocking=True).long()
        raw_long = raw_long_full
        query_rows = query_rows.to(device=device, dtype=torch.long, non_blocking=True)
        self_row = torch.arange(num_bg, device=device, dtype=torch.long)[:, None]

        full_to_struct = torch.full((bg_rows,), -1, device=device, dtype=torch.long)
        full_to_struct[query_rows] = torch.arange(num_bg, device=device, dtype=torch.long)
        raw_valid_global = (raw_long >= 0) & (raw_long < bg_rows)

        mapped = torch.full_like(raw_long, -1)
        mapped[raw_valid_global] = full_to_struct[raw_long[raw_valid_global]]

        valid = raw_valid_global & (mapped >= 0) & (mapped != self_row)
        safe = torch.where(valid, mapped, self_row.expand_as(mapped))
        neighbor_idx = torch.cat([self_row, safe], dim=1).contiguous()
        neighbor_mask = torch.cat([torch.ones((num_bg, 1), device=device, dtype=torch.bool), valid], dim=1).contiguous()
        return neighbor_idx, neighbor_mask

    def _build_rigid_in_struct_neighbors(
        self,
        *,
        route: RigidRoute,
        rigid_knn_idx: torch.Tensor,
        rigid_knn_row_ids: Optional[torch.Tensor],
        num_bg: int,
        num_rigid: int,
        k: int,
        min_valid_neighbors: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        num_in = int(route.S_in.numel())
        k_store = int(k) - 1

        if num_in <= 0:
            empty_idx = torch.empty((0, k), device=device, dtype=torch.long)
            empty_mask = torch.empty((0, k), device=device, dtype=torch.bool)
            return empty_idx, empty_mask, {
                "rigid_out_neighbor_ratio": 0.0,
                "rigid_self_only_ratio": 0.0,
                "rigid_valid_neighbor_mean": 0.0,
            }
        if int(rigid_knn_idx.shape[1]) != k_store:
            raise RuntimeError(f"rigid_knn_idx K_store mismatch: got {rigid_knn_idx.shape[1]}, required {k_store}")

        rigid_global_to_struct = torch.full((num_rigid,), -1, device=device, dtype=torch.long)
        rigid_in_struct_rows = num_bg + torch.arange(num_in, device=device, dtype=torch.long)
        rigid_global_to_struct[route.S_in] = rigid_in_struct_rows

        query_global = route.S_in.to(device=device, dtype=torch.long)
        self_row = rigid_global_to_struct[query_global][:, None]
        knn_dev = rigid_knn_idx.device
        knn_on_cpu = str(knn_dev.type) == "cpu"

        if rigid_knn_row_ids is None:
            if int(rigid_knn_idx.shape[0]) != int(num_rigid):
                raise RuntimeError(
                    "rigid_knn_idx row count mismatches current rigid node state and no rigid_knn_row_ids mapping is provided: "
                    f"knn_rows={rigid_knn_idx.shape[0]} num_rigid={num_rigid}. "
                    "This indicates KNN row-space mismatch (e.g. filtered knn table vs full node_state_rigid rows)."
                )
            query_knn_rows = query_global.to(
                device=knn_dev,
                dtype=torch.long,
                non_blocking=not knn_on_cpu,
            ).contiguous()
            query_has_row = torch.ones_like(query_global, dtype=torch.bool, device=device)
        else:
            row_ids = rigid_knn_row_ids.to(device=device, dtype=torch.long, non_blocking=True)
            if bool((row_ids < 0).any().item()) or bool((row_ids >= num_rigid).any().item()):
                raise RuntimeError(
                    f"rigid_knn_row_ids out of range [0, {num_rigid}). "
                    "KNN payload row mapping must reference current node_state_rigid row space."
                )
            global_to_knn = torch.full((num_rigid,), -1, device=device, dtype=torch.long)
            global_to_knn[row_ids] = torch.arange(int(row_ids.shape[0]), device=device, dtype=torch.long)
            query_knn_rows_raw = global_to_knn[query_global]
            query_has_row = query_knn_rows_raw >= 0
            if bool((~query_has_row).any().item()):
                missing = int((~query_has_row).sum().item())
                raise RuntimeError(
                    "rigid_knn_row_ids does not cover all rigid_in query rows. "
                    f"missing={missing} num_queries={int(query_global.shape[0])}."
                )
            query_knn_rows = query_knn_rows_raw.to(
                device=knn_dev,
                dtype=torch.long,
                non_blocking=not knn_on_cpu,
            ).contiguous()

        knn_rows = int(rigid_knn_idx.shape[0])
        if knn_rows <= 0:
            raise RuntimeError("Stage5_1 rigid_knn_idx has zero rows while rigid_in queries are present.")
        bad_low = query_knn_rows < 0
        bad_high = query_knn_rows >= knn_rows
        bad_mask = bad_low | bad_high
        query_row_valid = ~bad_mask
        query_has_row = query_has_row & query_row_valid.to(device=query_has_row.device)
        if bool(bad_mask.any().item()):
            qmin = int(query_knn_rows.min().item()) if int(query_knn_rows.numel()) > 0 else -1
            qmax = int(query_knn_rows.max().item()) if int(query_knn_rows.numel()) > 0 else -1
            bad = int(bad_mask.sum().item())
            bad_low_n = int(bad_low.sum().item())
            bad_high_n = int(bad_high.sum().item())
            qgmin = int(query_global.min().item()) if int(query_global.numel()) > 0 else -1
            qgmax = int(query_global.max().item()) if int(query_global.numel()) > 0 else -1
            raise RuntimeError(
                "Stage5_1 detected rigid query rows outside knn table range: "
                f"bad={bad} bad_low={bad_low_n} bad_high={bad_high_n} "
                f"qmin={qmin} qmax={qmax} query_global_min={qgmin} query_global_max={qgmax} "
                f"knn_rows={knn_rows} num_rigid={num_rigid} with_row_ids={rigid_knn_row_ids is not None}."
            )

        raw_global_full = rigid_knn_idx[query_knn_rows].to(device=device, non_blocking=True).long()
        raw_global = raw_global_full

        raw_valid_global = (raw_global >= 0) & (raw_global < num_rigid)
        raw_not_self = raw_global != query_global[:, None]

        mapped = torch.full_like(raw_global, -1)
        mapped[raw_valid_global] = rigid_global_to_struct[raw_global[raw_valid_global]]
        valid_in = raw_valid_global & raw_not_self & (mapped >= 0) & query_has_row[:, None]
        valid_count = valid_in.sum(dim=1)

        force_self_only = valid_count < int(min_valid_neighbors)
        valid_final = valid_in & (~force_self_only[:, None])
        safe = torch.where(valid_final, mapped, self_row.expand_as(mapped))

        neighbor_idx = torch.cat([self_row, safe], dim=1).contiguous()
        neighbor_mask = torch.cat(
            [torch.ones((num_in, 1), device=device, dtype=torch.bool), valid_final],
            dim=1,
        ).contiguous()

        raw_valid_nonself = raw_valid_global & raw_not_self & query_has_row[:, None]
        out_or_invalid = raw_valid_nonself & (mapped < 0)
        denom = raw_valid_nonself.sum().clamp(min=1)
        aux = {
            "rigid_out_neighbor_ratio": float(out_or_invalid.sum().detach().float().div(denom).item()),
            "rigid_self_only_ratio": float(force_self_only.detach().float().mean().item()),
            "rigid_valid_neighbor_mean": float(valid_count.detach().float().mean().item()),
        }
        return neighbor_idx, neighbor_mask, aux

    def _build_struct_knn_neighbors_bg_rigid_in(
        self,
        *,
        batch: Dict[str, Any],
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        num_bg: int,
        num_rigid_in: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        knn_cfg = self.stage5_1_knn_cfg
        k = int(self._require_key(knn_cfg, "k", "model.struct_decoder.knn_attention"))
        min_valid_neighbors = int(self._require_key(knn_cfg, "min_valid_neighbors", "model.struct_decoder.knn_attention"))
        bg_knn_idx, rigid_knn_idx, rigid_knn_row_ids = self._get_segment_knn_tensors(batch)

        bg_idx, bg_mask = self._build_bg_struct_neighbors(
            batch=batch,
            bg_knn_idx=bg_knn_idx,
            num_bg=num_bg,
            k=k,
            device=self.device,
        )

        if node_state_rigid is not None and num_rigid_in > 0:
            rigid_idx, rigid_mask, rigid_aux = self._build_rigid_in_struct_neighbors(
                route=route,
                rigid_knn_idx=rigid_knn_idx,
                rigid_knn_row_ids=rigid_knn_row_ids,
                num_bg=num_bg,
                num_rigid=int(node_state_rigid.means.shape[0]),
                k=k,
                min_valid_neighbors=min_valid_neighbors,
                device=self.device,
            )
        else:
            rigid_idx = torch.empty((0, k), device=self.device, dtype=torch.long)
            rigid_mask = torch.empty((0, k), device=self.device, dtype=torch.bool)
            rigid_aux = {
                "rigid_out_neighbor_ratio": 0.0,
                "rigid_self_only_ratio": 0.0,
                "rigid_valid_neighbor_mean": 0.0,
            }

        neighbor_idx = torch.cat([bg_idx, rigid_idx], dim=0).contiguous()
        neighbor_mask = torch.cat([bg_mask, rigid_mask], dim=0).contiguous()
        aux = {
            "stage5_1_knn_k": float(k),
            "stage5_1_knn_bg_valid_neighbor_mean": (
                float(bg_mask[:, 1:].sum(dim=1).float().mean().item()) if num_bg > 0 else 0.0
            ),
            "stage5_1_knn_rigid_valid_neighbor_mean": rigid_aux["rigid_valid_neighbor_mean"],
            "stage5_1_knn_rigid_out_neighbor_ratio": rigid_aux["rigid_out_neighbor_ratio"],
            "stage5_1_knn_rigid_self_only_ratio": rigid_aux["rigid_self_only_ratio"],
        }
        return neighbor_idx, neighbor_mask, aux

    def _compute_bg_rigid_in_gru_inputs(
        self,
        *,
        batch: Optional[Dict[str, Any]] = None,
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_rigid_S: Optional[torch.Tensor],
        node_state_distant: Optional[NodeStateDistant] = None,
        feat_2d_distant: Optional[torch.Tensor] = None,
        acc_w_distant: Optional[torch.Tensor] = None,
    ) -> BgRigidInGRUInputs:
        _ = node_state_distant
        _ = feat_2d_distant
        _ = acc_w_distant
        struct_in = self._build_struct_decoder_input_bg_rigid_in(
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_bg=feat_2d_bg,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_bg=acc_w_bg,
            acc_w_rigid_S=acc_w_rigid_S,
        )
        neighbor_idx, neighbor_mask, knn_aux = self._build_struct_knn_neighbors_bg_rigid_in(
            batch=batch if batch is not None else {},
            node_state_rigid=node_state_rigid,
            route=route,
            num_bg=int(struct_in.split_bg),
            num_rigid_in=int(struct_in.split_rigid_in),
        )
        struct_in.neighbor_idx = neighbor_idx
        struct_in.neighbor_mask = neighbor_mask
        struct_in.meta.update(knn_aux)

        batch_offsets = self._build_struct_batch_offsets(struct_in)
        struct_out = self.struct_decoder(
            struct_in,
            aabb_min=self.bbx_min,
            aabb_max=self.bbx_max,
            batch_offsets=batch_offsets,
        )

        feat_struct = struct_out.feat
        num_bg = int(struct_in.split_bg)
        num_rigid_in = int(struct_in.split_rigid_in)
        feat_bg_input = feat_struct[:num_bg]
        feat_rigid_in_input_all = feat_struct[num_bg : num_bg + num_rigid_in] if num_rigid_in > 0 else None

        aux = dict(struct_out.aux)
        aux.update(knn_aux)
        aux.update(
            {
                "stage5_1_struct_enabled": 1.0,
                "stage5_1_struct_num_bg": float(num_bg),
                "stage5_1_struct_num_rigid_in": float(num_rigid_in),
                "stage5_1_bg_struct_feat_norm": float(feat_bg_input.norm(dim=-1).mean().item()) if num_bg > 0 else 0.0,
                "stage5_1_rigid_in_struct_feat_norm": (
                    float(feat_rigid_in_input_all.norm(dim=-1).mean().item()) if feat_rigid_in_input_all is not None else 0.0
                ),
            }
        )
        return BgRigidInGRUInputs(
            feat_bg_input=feat_bg_input,
            feat_rigid_in_input_all=feat_rigid_in_input_all,
            aux=aux,
        )

    def _build_struct_decoder_input_bg_rigid_in(
        self,
        *,
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_rigid_S: Optional[torch.Tensor],
    ) -> StructDecoderInput:
        num_bg = int(node_state_bg.means.shape[0])
        num_rigid_in = int(route.S_in.numel())

        feat_2d_parts = [feat_2d_bg]
        acc_w_parts = [acc_w_bg]
        coords_parts = [node_state_bg.means]
        branch_id_bg = torch.zeros((num_bg,), dtype=torch.long, device=self.device)
        branch_ids = [branch_id_bg]

        if num_rigid_in > 0:
            if node_state_rigid is None:
                raise RuntimeError("Stage5_1 got route.S_in > 0 but node_state_rigid is None.")
            if feat_2d_rigid_S is None or acc_w_rigid_S is None:
                raise RuntimeError("Stage5_1 expected rigid source 2D/support features for S_in.")
            rows_rigid_in_in_s = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_2d_rigid_in = feat_2d_rigid_S[rows_rigid_in_in_s]
            acc_w_rigid_in = acc_w_rigid_S[rows_rigid_in_in_s]
            coords_rigid_in = route.means_world_S[route.inside_mask_S]
            feat_2d_parts.append(feat_2d_rigid_in)
            acc_w_parts.append(acc_w_rigid_in)
            coords_parts.append(coords_rigid_in)
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=self.device))

        feat_2d = torch.cat(feat_2d_parts, dim=0)
        acc_w = torch.cat(acc_w_parts, dim=0)
        coords = torch.cat(coords_parts, dim=0)
        branch_id = torch.cat(branch_ids, dim=0)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        params_rigid_in = None
        if num_rigid_in > 0:
            params_rigid_in = self._build_rigid_params_for_embed_source_world(
                node_state_rigid,
                source_frame_idx,
                route.S_in,
            )
        params_struct = cat_param_dict(params_bg, params_rigid_in)

        return StructDecoderInput(
            feat_2d=feat_2d,
            acc_w=acc_w,
            coords=coords,
            branch_id=branch_id,
            params_for_embed=params_struct,
            split_bg=num_bg,
            split_rigid_in=num_rigid_in,
            meta={
                "support_threshold_bg": float(self.bg_src_backproject_support_min),
                "support_threshold_rigid": float(self.rigid_src_backproject_support_min),
            },
        )


__all__ = ["MinimalStreetForwardStage5_1"]
