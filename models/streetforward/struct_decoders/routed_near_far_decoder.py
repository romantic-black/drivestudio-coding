from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.streetforward.struct_decoders.common import StructDecoderInput, StructDecoderOutput


class RoutedNearFarStructDecoder(nn.Module):
    def __init__(self, *, near_decoder: nn.Module, far_decoder: nn.Module) -> None:
        super().__init__()
        self.near_decoder = near_decoder
        self.far_decoder = far_decoder

    @staticmethod
    def _validate_near_input(x: StructDecoderInput) -> None:
        n = int(x.coords.shape[0])
        if int(x.split_bg + x.split_rigid_in) != n:
            raise ValueError("RoutedNearFarStructDecoder near split mismatch with total points.")
        if bool((x.branch_id < 0).any().item()) or bool((x.branch_id > 1).any().item()):
            raise ValueError("RoutedNearFarStructDecoder near branch_id must be in {0,1}.")

    @staticmethod
    def _validate_far_input(x: StructDecoderInput) -> None:
        n = int(x.coords.shape[0])
        if int(x.split_bg + x.split_rigid_in) != n:
            raise ValueError("RoutedNearFarStructDecoder far split mismatch with total points.")
        if bool((x.branch_id < 0).any().item()) or bool((x.branch_id > 1).any().item()):
            raise ValueError("RoutedNearFarStructDecoder far branch_id must be in far-local {0,1}.")

    def decode_near(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        self._validate_near_input(x)
        return self.near_decoder(
            x,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
        )

    def decode_far(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        self._validate_far_input(x)
        return self.far_decoder(
            x,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            batch_offsets=batch_offsets,
        )


__all__ = ["RoutedNearFarStructDecoder"]
