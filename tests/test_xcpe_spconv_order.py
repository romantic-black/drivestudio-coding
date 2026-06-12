from __future__ import annotations

import torch

from models.streetforward.struct_decoders.xcpe_decoder import _XCPEResidualLayer


def test_xcpe_remaps_spconv_features_to_input_voxel_order() -> None:
    input_indices = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=torch.int32,
    )
    order = torch.tensor([2, 0, 1])
    out_indices = input_indices[order]
    out_features = torch.tensor(
        [
            [20.0, 21.0],
            [0.0, 1.0],
            [10.0, 11.0],
        ]
    )

    remapped = _XCPEResidualLayer._remap_features_to_input_order(
        features=out_features,
        out_indices_bzyx=out_indices,
        input_indices_bzyx=input_indices,
        spatial_shape_zyx=torch.tensor([2, 2, 2]),
    )

    expected = torch.tensor(
        [
            [0.0, 1.0],
            [10.0, 11.0],
            [20.0, 21.0],
        ]
    )
    assert torch.allclose(remapped, expected)
