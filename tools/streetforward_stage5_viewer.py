from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import viser
from nerfview import CameraState, Viewer


class StreetForwardStage5Viewer(Viewer):
    @staticmethod
    def _axis_to_unit_vec(axis: str) -> np.ndarray:
        amap = {
            "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        }
        key = str(axis).strip().lower()
        if key not in amap:
            raise ValueError("demo.viewer.*_up_direction must be one of: +x, -x, +y, -y, +z, -z")
        return amap[key]

    def __init__(
        self,
        *,
        server: viser.ViserServer,
        controller: Any,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.controller = controller
        self._demo_tab_handles: Dict[str, Any] = {}
        self._suspend_scope_callbacks = False
        viewer_cfg = (self.controller.cfg.get("demo", {}) or {}).get("viewer", {}) or {}
        scene_up_axis = str(viewer_cfg.get("scene_up_direction", "+y")).strip().lower()
        camera_up_axis = str(viewer_cfg.get("camera_up_direction", scene_up_axis)).strip().lower()
        self._camera_up_lock_enable = bool(viewer_cfg.get("lock_camera_up_direction", True))
        self._camera_up_lock_cos_threshold = float(viewer_cfg.get("camera_up_lock_cos_threshold", 0.9995))
        self._camera_up_lock_cos_threshold = float(np.clip(self._camera_up_lock_cos_threshold, -1.0, 1.0))
        self._camera_up_world = self._axis_to_unit_vec(camera_up_axis)
        self._scene_up_axis = scene_up_axis
        # Keep scene up and initial camera up explicit to stabilize orbit semantics
        # (horizontal drag -> yaw around a fixed up axis).
        server.scene.set_up_direction(self._scene_up_axis)
        server.initial_camera.up = tuple(float(x) for x in self._camera_up_world.tolist())
        if self._camera_up_lock_enable:
            up_target = self._camera_up_world.copy()
            cos_thr = float(self._camera_up_lock_cos_threshold)

            def _on_client_connect(client: viser.ClientHandle) -> None:
                @client.camera.on_update
                def _lock_camera_up(cam: Any) -> None:
                    up = np.asarray(cam.up_direction, dtype=np.float64)
                    n = float(np.linalg.norm(up))
                    if n <= 1e-8:
                        return
                    up = up / n
                    if float(np.dot(up, up_target)) < cos_thr:
                        cam.up_direction = up_target

            server.on_client_connect(_on_client_connect)
        # nerfview.Viewer expects this attribute in training mode while
        # populating the rendering tab (used as extra disable-able handles).
        self._training_tab_handles: Dict[str, Any] = {}
        super().__init__(
            server=server,
            render_fn=self._render_from_state,
            output_dir=output_dir,
            mode="training",
        )
        self.server.gui.set_panel_label("StreetForward Stage5 Demo")
        self.refresh_panel_state()

    def _init_training_tab(self):
        self._demo_tab_handles = {}
        self._training_tab_handles = {}
        self._demo_folder = self.server.gui.add_folder("Inference")

    def _populate_training_tab(self):
        viewer_cfg = self.controller.cfg.get("demo", {}).get("viewer", {}) or {}
        scene_options = tuple(str(x) for x in self.controller.list_scene_ids())
        if len(scene_options) == 0:
            scene_options = ("-1",)
        initial_scene = int(scene_options[0])
        segment_options = tuple(str(x) for x in self.controller.list_segment_ids(initial_scene))
        if len(segment_options) == 0:
            segment_options = ("-1",)
        initial_stats = self.controller.display.last_stats or {}
        initial_source_frame = int(initial_stats.get("source_frame_idx", -1))
        with self._demo_folder:
            stage_text = self.server.gui.add_text("Stage", initial_value="", disabled=True)
            scene_id = self.server.gui.add_number("Scene ID", initial_value=-1, disabled=True)
            segment_id = self.server.gui.add_number("Segment ID", initial_value=-1, disabled=True)
            scene_select = self.server.gui.add_dropdown(
                "Scene",
                options=scene_options,
                initial_value=scene_options[0],
            )
            segment_select = self.server.gui.add_dropdown(
                "Segment",
                options=segment_options,
                initial_value=segment_options[0],
            )
            global_step = self.server.gui.add_number("Global Step", initial_value=0, disabled=True)
            block_global = self.server.gui.add_number("Block Global", initial_value=-1, disabled=True)
            segment_step = self.server.gui.add_number("Segment Step", initial_value=-1, disabled=True)
            source_refs = self.server.gui.add_text("Source Refs", initial_value="[]", disabled=True)
            target_refs = self.server.gui.add_text("Target Refs", initial_value="[]", disabled=True)
            last_event = self.server.gui.add_text("Last Event", initial_value="", disabled=True)
            num_bg_update = self.server.gui.add_number("num_bg_update", initial_value=0, disabled=True)
            num_distant_update = self.server.gui.add_number("num_distant_update", initial_value=0, disabled=True)
            num_rigid_update = self.server.gui.add_number("num_rigid_update", initial_value=0, disabled=True)
            loss_val = self.server.gui.add_number("loss", initial_value=0.0, disabled=True)
            trained_steps_total = self.server.gui.add_number("Trained Steps Total", initial_value=0, disabled=True)
            trained_steps_since_reset = self.server.gui.add_number(
                "Trained Steps Since Param Reset",
                initial_value=0,
                disabled=True,
            )

            prev_scene = self.server.gui.add_button("Prev Scene")
            next_scene = self.server.gui.add_button("Next Scene")
            prev_segment = self.server.gui.add_button("Prev Segment")
            next_segment = self.server.gui.add_button("Next Segment")
            prev_block = self.server.gui.add_button("Prev Block")
            next_step = self.server.gui.add_button("Next Step", color="blue")
            next_block = self.server.gui.add_button("Next Block")
            reset_state = self.server.gui.add_button("Reset Current Segment State")
            reset_all_state = self.server.gui.add_button("Reset All Demo State")
            next_episode_reset = self.server.gui.add_button("Next Episode + Reset Segment")
            reset_train_params = self.server.gui.add_button("Reset Training Parameters")
            refresh_only = self.server.gui.add_button("Refresh Panel")
            show_bg = self.server.gui.add_checkbox("Show BG", initial_value=bool(viewer_cfg.get("show_bg", True)))
            show_distant = self.server.gui.add_checkbox(
                "Show Distant",
                initial_value=bool(viewer_cfg.get("show_distant", True)),
            )
            show_rigid = self.server.gui.add_checkbox(
                "Show Rigid",
                initial_value=bool(viewer_cfg.get("show_rigid", False)),
            )
            rigid_frame = self.server.gui.add_number("Rigid Frame", initial_value=int(initial_source_frame))
            lock_rigid_frame = self.server.gui.add_checkbox("Lock Rigid Frame", initial_value=False)

            def _refresh_scope_dropdowns_from_stats() -> None:
                stats = self.controller.display.last_stats or {}
                cur_scene = int(stats.get("scene_id", -1))
                cur_segment = int(stats.get("segment_id", -1))
                scenes = tuple(str(x) for x in self.controller.list_scene_ids())
                if len(scenes) == 0:
                    scenes = ("-1",)
                segments = tuple(str(x) for x in self.controller.list_segment_ids(cur_scene))
                if len(segments) == 0:
                    segments = ("-1",)
                self._suspend_scope_callbacks = True
                scene_select.options = scenes
                if str(cur_scene) in scenes:
                    scene_select.value = str(cur_scene)
                else:
                    scene_select.value = scenes[0]
                segment_select.options = segments
                if str(cur_segment) in segments:
                    segment_select.value = str(cur_segment)
                else:
                    segment_select.value = segments[0]
                self._suspend_scope_callbacks = False

            @next_step.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                next_step.disabled = True
                next_block.disabled = True
                try:
                    self.controller.step_current_block_once()
                    self.rerender(None)
                    _refresh_scope_dropdowns_from_stats()
                    self.refresh_panel_state()
                finally:
                    next_step.disabled = False
                    next_block.disabled = False

            @prev_block.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.prev_block()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @next_block.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.next_block()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @prev_scene.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.prev_scene()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @next_scene.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.next_scene()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @prev_segment.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.prev_segment()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @next_segment.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.next_segment()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @scene_select.on_update
            def _(_) -> None:
                if self.controller.busy or self._suspend_scope_callbacks:
                    return
                scene_val = int(scene_select.value)
                segments = self.controller.list_segment_ids(scene_val)
                target_segment = int(segments[0]) if len(segments) > 0 else -1
                if target_segment < 0:
                    return
                self.controller.set_scope(scene_val, target_segment)
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @segment_select.on_update
            def _(_) -> None:
                if self.controller.busy or self._suspend_scope_callbacks:
                    return
                scene_val = int(scene_select.value)
                segment_val = int(segment_select.value)
                self.controller.set_scope(scene_val, segment_val)
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @reset_state.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.reset_current_segment_state()
                self.rerender(None)
                self.refresh_panel_state()

            @reset_all_state.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.reset_all_demo_state()
                self.rerender(None)
                self.refresh_panel_state()

            @next_episode_reset.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.new_episode_and_reset_segment_state()
                self.rerender(None)
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @reset_train_params.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.reset_training_parameters()
                self.rerender(None)
                self.refresh_panel_state()

            @refresh_only.on_click
            def _(_) -> None:
                _refresh_scope_dropdowns_from_stats()
                self.refresh_panel_state()

            @show_distant.on_update
            def _(_) -> None:
                self.rerender(None)

            @show_bg.on_update
            def _(_) -> None:
                self.rerender(None)

            @show_rigid.on_update
            def _(_) -> None:
                self.rerender(None)

            @rigid_frame.on_update
            def _(_) -> None:
                self.rerender(None)

            @lock_rigid_frame.on_update
            def _(_) -> None:
                if not bool(lock_rigid_frame.value):
                    stats = self.controller.display.last_stats or {}
                    rigid_frame.value = int(stats.get("source_frame_idx", -1))
                self.rerender(None)

        handles = {
            "stage_text": stage_text,
            "scene_id": scene_id,
            "segment_id": segment_id,
            "scene_select": scene_select,
            "segment_select": segment_select,
            "global_step": global_step,
            "block_global": block_global,
            "segment_step": segment_step,
            "source_refs": source_refs,
            "target_refs": target_refs,
            "last_event": last_event,
            "num_bg_update": num_bg_update,
            "num_distant_update": num_distant_update,
            "num_rigid_update": num_rigid_update,
            "loss": loss_val,
            "trained_steps_total": trained_steps_total,
            "trained_steps_since_reset": trained_steps_since_reset,
            "show_bg": show_bg,
            "show_distant": show_distant,
            "show_rigid": show_rigid,
            "rigid_frame": rigid_frame,
            "lock_rigid_frame": lock_rigid_frame,
        }
        self._demo_tab_handles = handles
        self._training_tab_handles = handles

    def _render_from_state(self, camera_state: CameraState, img_wh: Tuple[int, int]) -> np.ndarray:
        if not self._demo_tab_handles:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)
        show_bg = bool(self._demo_tab_handles["show_bg"].value)
        show_distant = bool(self._demo_tab_handles["show_distant"].value)
        show_rigid = bool(self._demo_tab_handles["show_rigid"].value)
        rigid_frame_value = int(self._demo_tab_handles["rigid_frame"].value)
        rigid_frame_idx = rigid_frame_value if rigid_frame_value >= 0 else None
        return self.controller.render(
            camera_state,
            img_wh,
            show_bg=show_bg,
            show_distant=show_distant,
            show_rigid=show_rigid,
            rigid_frame_idx=rigid_frame_idx,
        )

    def refresh_panel_state(self) -> None:
        if not self._demo_tab_handles:
            return
        stats = self.controller.display.last_stats or {}
        self._demo_tab_handles["stage_text"].value = str(stats.get("stage", ""))
        self._demo_tab_handles["scene_id"].value = int(stats.get("scene_id", -1))
        self._demo_tab_handles["segment_id"].value = int(stats.get("segment_id", -1))
        self._demo_tab_handles["global_step"].value = int(stats.get("global_step", 0))
        self._demo_tab_handles["block_global"].value = int(stats.get("block_idx_global", -1))
        self._demo_tab_handles["segment_step"].value = int(stats.get("segment_local_step", -1))
        self._demo_tab_handles["source_refs"].value = str(list(stats.get("source_image_refs", [])))
        self._demo_tab_handles["target_refs"].value = str(list(stats.get("target_image_refs", [])))
        self._demo_tab_handles["last_event"].value = str(stats.get("last_event_type", ""))
        self._demo_tab_handles["num_bg_update"].value = int(stats.get("num_bg_update", 0))
        self._demo_tab_handles["num_distant_update"].value = int(stats.get("num_distant_update", 0))
        self._demo_tab_handles["num_rigid_update"].value = int(stats.get("num_rigid_update", 0))
        self._demo_tab_handles["loss"].value = float(stats.get("loss", 0.0))
        self._demo_tab_handles["trained_steps_total"].value = int(stats.get("trained_steps_total", 0))
        self._demo_tab_handles["trained_steps_since_reset"].value = int(
            stats.get("trained_steps_since_param_reset", 0)
        )
        if (
            "rigid_frame" in self._demo_tab_handles
            and "lock_rigid_frame" in self._demo_tab_handles
            and not bool(self._demo_tab_handles["lock_rigid_frame"].value)
        ):
            self._demo_tab_handles["rigid_frame"].value = int(stats.get("source_frame_idx", -1))
        scene_val = int(stats.get("scene_id", -1))
        segment_val = int(stats.get("segment_id", -1))
        if "scene_select" in self._demo_tab_handles and "segment_select" in self._demo_tab_handles:
            scenes = tuple(str(x) for x in self.controller.list_scene_ids())
            if len(scenes) == 0:
                scenes = ("-1",)
            segments = tuple(str(x) for x in self.controller.list_segment_ids(scene_val))
            if len(segments) == 0:
                segments = ("-1",)
            self._suspend_scope_callbacks = True
            self._demo_tab_handles["scene_select"].options = scenes
            self._demo_tab_handles["scene_select"].value = str(scene_val) if str(scene_val) in scenes else scenes[0]
            self._demo_tab_handles["segment_select"].options = segments
            self._demo_tab_handles["segment_select"].value = (
                str(segment_val) if str(segment_val) in segments else segments[0]
            )
            self._suspend_scope_callbacks = False
