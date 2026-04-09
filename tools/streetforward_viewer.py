from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import viser
from nerfview import CameraState, Viewer

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ImportError:
    _gsplat_rasterization = None


def _to_numpy_uint8(rgb: torch.Tensor) -> np.ndarray:
    rgb01 = torch.clamp(rgb, 0.0, 1.0).detach().cpu().numpy()
    return (rgb01 * 255.0).astype(np.uint8)


class StreetForwardViewer(Viewer):
    def __init__(
        self,
        *,
        server: viser.ViserServer,
        controller: Any,
        output_dir: Optional[Path] = None,
    ):
        if _gsplat_rasterization is None:
            raise ImportError("gsplat is not available; viewer demo requires gsplat.rendering.rasterization.")
        self.controller = controller
        self._aabb_handle = None
        self._aabb_path = "/streetforward/segment_aabb"
        self._training_tab_handles: Dict[str, Any] = {}
        super().__init__(
            server=server,
            render_fn=self._render_from_snapshot,
            output_dir=output_dir,
            mode="training",
        )
        self.server.gui.set_panel_label("StreetForward Block Demo")
        self.refresh_panel_state()
        self._update_segment_aabb_visual()

    def _init_training_tab(self):
        self._training_tab_handles = {}
        self._training_folder = self.server.gui.add_folder("Training")

    def _populate_training_tab(self):
        with self._training_folder:
            scene_id = self.server.gui.add_number("Scene ID", initial_value=-1, disabled=True)
            segment_id = self.server.gui.add_number("Segment ID", initial_value=-1, disabled=True)
            block_global = self.server.gui.add_number("Block Global", initial_value=-1, disabled=True)
            segment_step = self.server.gui.add_number("Segment Step", initial_value=-1, disabled=True)
            source_ref = self.server.gui.add_text("Source Image Ref", initial_value="(-1,-1)", disabled=True)
            target_refs = self.server.gui.add_text("Target Image Refs", initial_value="[]", disabled=True)
            busy = self.server.gui.add_text("Busy", initial_value="false", disabled=True)
            num_bg_update = self.server.gui.add_number("Last num_bg_update", initial_value=0, disabled=True)
            num_distant_update = self.server.gui.add_number("Last num_distant_update", initial_value=0, disabled=True)
            num_sky_update = self.server.gui.add_number("Last num_sky_update", initial_value=0, disabled=True)
            num_rigid_update = self.server.gui.add_number("Last num_rigid_update", initial_value=0, disabled=True)

            train_next_block = self.server.gui.add_button("Train Next Block", color="blue")
            train_next_5 = self.server.gui.add_button("Train Next 5 Blocks")
            refresh_snapshot = self.server.gui.add_button("Refresh Snapshot")
            reset_runtime = self.server.gui.add_button("Reset Runtime")
            export_3dgs = self.server.gui.add_button("Export 3DGS")
            auto_refresh = self.server.gui.add_checkbox("Auto Refresh After Block", initial_value=True)
            show_distant = self.server.gui.add_checkbox("Show Distant", initial_value=True)
            show_segment_aabb = self.server.gui.add_checkbox("Show Segment AABB", initial_value=True)
            show_stats = self.server.gui.add_checkbox("Show Stats", initial_value=True)

            @train_next_block.on_click
            def _(_) -> None:
                self.controller.train_next_block(1)
                if auto_refresh.value:
                    self.rerender(None)
                self.refresh_panel_state()
                self._update_segment_aabb_visual()

            @train_next_5.on_click
            def _(_) -> None:
                self.controller.train_next_block(5)
                if auto_refresh.value:
                    self.rerender(None)
                self.refresh_panel_state()
                self._update_segment_aabb_visual()

            @refresh_snapshot.on_click
            def _(_) -> None:
                self.controller.build_or_refresh_snapshot()
                self.rerender(None)
                self.refresh_panel_state()
                self._update_segment_aabb_visual()

            @reset_runtime.on_click
            def _(_) -> None:
                self.controller.reset_runtime_to_segment_init()
                self.rerender(None)
                self.refresh_panel_state()
                self._update_segment_aabb_visual()

            @export_3dgs.on_click
            def _(_) -> None:
                self.controller.export_current_snapshot("outputs/streetforward_viewer_snapshot.pt")
                self.refresh_panel_state()

            @auto_refresh.on_update
            def _(_) -> None:
                self.controller.auto_refresh_after_block = bool(auto_refresh.value)

            @show_distant.on_update
            def _(_) -> None:
                self.rerender(None)

            @show_segment_aabb.on_update
            def _(_) -> None:
                self._update_segment_aabb_visual()

            @show_stats.on_update
            def _(_) -> None:
                visible = bool(show_stats.value)
                num_bg_update.visible = visible
                num_distant_update.visible = visible
                num_sky_update.visible = visible
                num_rigid_update.visible = visible

        self._training_tab_handles = {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "block_global": block_global,
            "segment_step": segment_step,
            "source_ref": source_ref,
            "target_refs": target_refs,
            "busy": busy,
            "num_bg_update": num_bg_update,
            "num_distant_update": num_distant_update,
            "num_sky_update": num_sky_update,
            "num_rigid_update": num_rigid_update,
            "show_distant": show_distant,
            "show_segment_aabb": show_segment_aabb,
        }

    def _build_aabb_mesh(self, aabb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mins = aabb[0]
        maxs = aabb[1]
        verts = np.array(
            [
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int32,
        )
        return verts, faces

    def _update_segment_aabb_visual(self) -> None:
        show = bool(self._training_tab_handles.get("show_segment_aabb").value) if self._training_tab_handles else False
        snap = self.controller.display.current_snapshot
        if snap is None or "segment_aabb" not in snap:
            if self._aabb_handle is not None:
                self._aabb_handle.remove()
                self._aabb_handle = None
            return

        aabb = snap["segment_aabb"]
        if torch.is_tensor(aabb):
            aabb_np = aabb.detach().cpu().numpy()
        else:
            aabb_np = np.asarray(aabb, dtype=np.float32)
        if aabb_np.shape != (2, 3):
            return

        verts, faces = self._build_aabb_mesh(aabb_np)
        if self._aabb_handle is not None:
            self._aabb_handle.remove()
        self._aabb_handle = self.server.scene.add_mesh_simple(
            self._aabb_path,
            vertices=verts,
            faces=faces,
            color=(255, 180, 60),
            wireframe=True,
            opacity=1.0,
            visible=show,
        )

    @torch.no_grad()
    def _render_from_snapshot(self, camera_state: CameraState, img_wh: Tuple[int, int]):
        snap = self.controller.display.current_snapshot
        if snap is None:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)
        gs_state = snap.get("gs_state") or {}
        branches = (gs_state.get("branches") or {})
        show_distant = bool(self._training_tab_handles.get("show_distant").value) if self._training_tab_handles else True

        branch_names = ["bg"] + (["distant"] if show_distant else [])
        means_list: List[torch.Tensor] = []
        scales_list: List[torch.Tensor] = []
        quats_list: List[torch.Tensor] = []
        opacities_list: List[torch.Tensor] = []
        colors_list: List[torch.Tensor] = []

        device = self.controller.device
        for name in branch_names:
            b = branches.get(name)
            if not isinstance(b, dict):
                continue
            if b.get("means") is None:
                continue
            means = b["means"].to(device)
            scales = torch.exp(b["scales_log"].to(device))
            quats = b["quats"].to(device)
            opacities = torch.sigmoid(b["opacity_logit"].to(device)).squeeze(-1)
            sh_dc = b["sh_dc"].to(device)
            sh_rest = b["sh_rest"].to(device)
            colors = torch.cat([sh_dc[:, None, :], sh_rest], dim=1)
            means_list.append(means)
            scales_list.append(scales)
            quats_list.append(quats)
            opacities_list.append(opacities)
            colors_list.append(colors)

        w, h = img_wh
        if len(means_list) == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        means = torch.cat(means_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        quats = torch.cat(quats_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        colors = torch.cat(colors_list, dim=0)

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        k = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
        render_colors, _, _ = _gsplat_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w)[None, ...],
            Ks=k[None, ...],
            width=int(w),
            height=int(h),
            packed=False,
            rasterize_mode="antialiased",
        )
        return _to_numpy_uint8(render_colors[0])

    def refresh_panel_state(self) -> None:
        if not self._training_tab_handles:
            return
        info = self.controller.display.current_scheduler_info or {}
        snap = self.controller.display.current_snapshot or {}
        stats = snap.get("stats") or {}
        source_ref = snap.get("source_image_ref", (-1, -1))
        target_refs = snap.get("target_image_refs", [])

        self._training_tab_handles["scene_id"].value = int(info.get("scene_id", -1))
        self._training_tab_handles["segment_id"].value = int(info.get("segment_id", -1))
        self._training_tab_handles["block_global"].value = int(snap.get("block_idx_global", info.get("block_idx_global", -1)))
        self._training_tab_handles["segment_step"].value = int(
            snap.get("segment_local_step", info.get("segment_local_step", -1))
        )
        self._training_tab_handles["source_ref"].value = str(tuple(source_ref))
        self._training_tab_handles["target_refs"].value = str([tuple(x) for x in target_refs])
        self._training_tab_handles["busy"].value = "true" if self.controller.busy else "false"
        self._training_tab_handles["num_bg_update"].value = int(stats.get("num_bg_update", 0))
        self._training_tab_handles["num_distant_update"].value = int(stats.get("num_distant_update", 0))
        self._training_tab_handles["num_sky_update"].value = int(stats.get("num_sky_update", 0))
        self._training_tab_handles["num_rigid_update"].value = int(stats.get("num_rigid_update", 0))

