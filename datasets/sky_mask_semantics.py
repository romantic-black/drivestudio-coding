"""Sky mask tensor semantics before/after normalization.

`pixel_source.load_sky_masks` uses PNG ``> 0`` as float 1. Depending on preprocessing,
that may mean non-sky (common) or sky. MultiSceneDataset normalizes to a single convention:

  **1 = sky, 0 = non-sky** (float in [0, 1], same shape as image).

See docs/dataloader/Sky_Mask_Semantics_One_Is_Sky_Refactor.md.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

SkyMaskLoaderSemantics = Literal["one_is_sky", "one_is_non_sky"]


def normalize_sky_mask_to_one_is_sky(
    sky_mask: Tensor,
    loader_semantics: SkyMaskLoaderSemantics,
) -> Tensor:
    """Map loader tensor to canonical **1=sky, 0=non-sky**."""
    x = sky_mask.float()
    if loader_semantics == "one_is_sky":
        return x
    if loader_semantics == "one_is_non_sky":
        return 1.0 - x
    raise ValueError(
        f"loader_semantics must be one_is_sky|one_is_non_sky, got {loader_semantics!r}"
    )


def parse_sky_mask_semantics_from_data_cfg(data_cfg) -> Optional[SkyMaskLoaderSemantics]:
    """
    Read ``data.sky_mask_semantics`` (required when ``pixel_source.load_sky_mask`` is true).
    Returns None when sky masks are not loaded.
    """
    from omegaconf import OmegaConf

    load_sm = OmegaConf.select(data_cfg, "pixel_source.load_sky_mask")
    if load_sm is None or not bool(load_sm):
        return None
    raw = OmegaConf.select(data_cfg, "sky_mask_semantics")
    if raw is None:
        raise ValueError(
            "data.sky_mask_semantics is required when pixel_source.load_sky_mask is true. "
            "Use one_is_non_sky if PNG nonzero means non-sky (typical nuScenes/EVolSplat), "
            "or one_is_sky if nonzero means sky."
        )
    s = str(raw).strip()
    if s not in ("one_is_sky", "one_is_non_sky"):
        raise ValueError(
            f"data.sky_mask_semantics must be one_is_sky or one_is_non_sky, got {raw!r}"
        )
    return s  # type: ignore[return-value]
