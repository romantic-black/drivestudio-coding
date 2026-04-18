from __future__ import annotations

import torch

from models.feature_extractors.alpha_t_extractor import AlphaTWeightExtractor


def test_render_and_backproject_streaming_pairs_total_and_after_mask_are_distinct():
    extractor = AlphaTWeightExtractor.__new__(AlphaTWeightExtractor)
    extractor.tile_size = 16
    extractor.sh_degree = 0

    class _DummyCam:
        def __init__(self):
            self.camtoworlds = torch.eye(4).unsqueeze(0)
            self.K = torch.eye(3).unsqueeze(0)

    cams = [_DummyCam()]

    def _fake_resolve_intrinsics(cam):
        return cam.K

    def _fake_renderer(**kwargs):
        del kwargs
        return None, None, {"means2d": torch.zeros(1, 2), "tile_size": 16}

    def _fake_extract_single_weight(meta, height, width, pair_valid_mask=None):
        del meta, height, width, pair_valid_mask
        return {
            "gaussian_ids": torch.tensor([0, 0, 0], dtype=torch.long),
            "pixel_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "weights": torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        }

    class _DummyBackprojector:
        eps = 1e-8
        weight_threshold = 0.0

        @staticmethod
        def backproject_single_view(
            features_2d,
            weight_info,
            height,
            width,
            num_gaussians,
            return_support_weight=False,
            return_debug_stats=False,
        ):
            del features_2d, height, width
            kept = int(weight_info["gaussian_ids"].numel())
            feat_sum = torch.zeros(num_gaussians, 2, dtype=torch.float32)
            weight_sum_feature = torch.ones(num_gaussians, dtype=torch.float32)
            stats = {"pairs_after_threshold": kept}
            if return_support_weight:
                weight_sum_support = torch.full((num_gaussians,), float(kept), dtype=torch.float32)
                if return_debug_stats:
                    return feat_sum, weight_sum_feature, weight_sum_support, stats
                return feat_sum, weight_sum_feature, weight_sum_support
            if return_debug_stats:
                return feat_sum, weight_sum_feature, stats
            return feat_sum, weight_sum_feature

    extractor._resolve_intrinsics = _fake_resolve_intrinsics
    extractor.renderer = _fake_renderer
    extractor.extract_single_weight = _fake_extract_single_weight

    _, _, stats = extractor.render_and_backproject_streaming(
        gaussians={
            "means": torch.zeros(1, 3),
            "quats": torch.zeros(1, 4),
            "scales": torch.ones(1, 3),
            "opacities": torch.ones(1),
            "colors": torch.zeros(1, 3),
        },
        cameras=cams,
        features_2d=torch.zeros(1, 2, 2, 2),
        height=2,
        width=2,
        num_gaussians=1,
        backprojector=_DummyBackprojector(),
        source_pair_valid_mask=torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool),
        return_accumulated_weights=True,
        return_debug_stats=True,
    )
    assert stats["pairs_total"] == 3
    assert stats["pairs_after_mask"] == 2
