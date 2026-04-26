from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import viser
from nerfview import CameraState, Viewer


class StreetForwardStage5Viewer(Viewer):
    def __init__(
        self,
        *,
        server: viser.ViserServer,
        controller: Any,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.controller = controller
        self._demo_tab_handles: Dict[str, Any] = {}
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
        self._demo_folder = self.server.gui.add_folder("Inference")

    def _populate_training_tab(self):
        with self._demo_folder:
            stage_text = self.server.gui.add_text("Stage", initial_value="", disabled=True)
            scene_id = self.server.gui.add_number("Scene ID", initial_value=-1, disabled=True)
            segment_id = self.server.gui.add_number("Segment ID", initial_value=-1, disabled=True)
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

            next_step = self.server.gui.add_button("Next Step", color="blue")
            next_block = self.server.gui.add_button("Next Block")
            reset_state = self.server.gui.add_button("Reset Scene State")
            refresh_only = self.server.gui.add_button("Refresh Panel")
            show_distant = self.server.gui.add_checkbox("Show Distant", initial_value=True)

            @next_step.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                next_step.disabled = True
                next_block.disabled = True
                try:
                    self.controller.step_once()
                    self.rerender(None)
                    self.refresh_panel_state()
                finally:
                    next_step.disabled = False
                    next_block.disabled = False

            @next_block.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                next_step.disabled = True
                next_block.disabled = True
                try:
                    self.controller.step_block()
                    self.rerender(None)
                    self.refresh_panel_state()
                finally:
                    next_step.disabled = False
                    next_block.disabled = False

            @reset_state.on_click
            def _(_) -> None:
                if self.controller.busy:
                    return
                self.controller.reset_current_scene_state()
                self.rerender(None)
                self.refresh_panel_state()

            @refresh_only.on_click
            def _(_) -> None:
                self.refresh_panel_state()

            @show_distant.on_update
            def _(_) -> None:
                self.rerender(None)

        self._demo_tab_handles = {
            "stage_text": stage_text,
            "scene_id": scene_id,
            "segment_id": segment_id,
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
            "show_distant": show_distant,
        }

    def _render_from_state(self, camera_state: CameraState, img_wh: Tuple[int, int]) -> np.ndarray:
        if not self._demo_tab_handles:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)
        show_distant = bool(self._demo_tab_handles["show_distant"].value)
        return self.controller.render(camera_state, img_wh, show_distant=show_distant)

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

