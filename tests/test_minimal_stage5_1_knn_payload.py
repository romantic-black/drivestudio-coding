from __future__ import annotations

import numpy as np

from models.streetforward.minimal_trainer_stage5_1 import MinimalStreetForwardStage5_1


def _make_trainer_stub() -> MinimalStreetForwardStage5_1:
    trainer = MinimalStreetForwardStage5_1.__new__(MinimalStreetForwardStage5_1)
    trainer.stage5_1_knn_cfg = {"neighbor_policy": "fixed_cached"}
    return trainer


def test_stage5_1_fixed_cached_accepts_implicit_full_row_space_mapping() -> None:
    trainer = _make_trainer_stub()
    batch = {
        "knn_struct_neighbors": {
            "bg_knn_idx": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            "rigid_knn_idx": np.asarray([[0, 1], [1, 0], [2, 1]], dtype=np.int64),
        }
    }
    bg_knn_idx, rigid_knn_idx, rigid_knn_row_ids = trainer._get_segment_knn_tensors(batch)
    assert tuple(bg_knn_idx.shape) == (2, 2)
    assert tuple(rigid_knn_idx.shape) == (3, 2)
    assert rigid_knn_row_ids is None


def test_stage5_1_fixed_cached_accepts_complete_row_space_metadata() -> None:
    trainer = _make_trainer_stub()
    batch = {
        "knn_struct_neighbors": {
            "bg_knn_idx": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            "rigid_knn_idx": np.asarray([[0, 1], [1, 0], [2, 1]], dtype=np.int64),
            "rigid_knn_row_ids": np.asarray([0, 1, 2], dtype=np.int64),
            "rigid_instance_intids": np.asarray([9], dtype=np.int64),
            "rigid_instance_offsets": np.asarray([0, 3], dtype=np.int64),
        }
    }
    bg_knn_idx, rigid_knn_idx, rigid_knn_row_ids = trainer._get_segment_knn_tensors(batch)
    assert tuple(bg_knn_idx.shape) == (2, 2)
    assert tuple(rigid_knn_idx.shape) == (3, 2)
    assert rigid_knn_row_ids is not None
    assert tuple(rigid_knn_row_ids.shape) == (3,)
