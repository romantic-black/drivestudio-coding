from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torchsparse import nn as spnn
from torchsparse.tensor import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

TensorLike = Union[torch.Tensor, np.ndarray]


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class FeatureNet(nn.Module):
    """2D feature pyramid network used by EVolSplat."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1),
            ConvBnReLU(8, 8, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2),
            ConvBnReLU(16, 16, 3, 1, 1),
            ConvBnReLU(16, 16, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2),
            ConvBnReLU(32, 32, 3, 1, 1),
            ConvBnReLU(32, 32, 3, 1, 1),
        )

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x: torch.Tensor):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)
        return [feat2, feat1, feat0]


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc: int, outc: int, ks: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.net(x)


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc: int, outc: int, ks: int = 3, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.net(x)


class SparseResidualBlock(nn.Module):
    def __init__(self, inc: int, outc: int, ks: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            spnn.BatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.relu(self.net(x) + self.downsample(x))


class SparseCostRegNet(nn.Module):
    """Sparse 3D UNet style architecture used for cost regularization."""

    def __init__(self, d_in: int, d_out: int = 8):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)
        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)
        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)
        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)
        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)
        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)
        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

    def forward(self, x: SparseTensor) -> torch.Tensor:
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F


def sparse_to_dense_volume(
    sparse_tensor: Union[SparseTensor, torch.Tensor],
    coords: torch.Tensor,
    vol_dim: Union[torch.Tensor, np.ndarray, Tuple[int, int, int], list],
    default_val: float = 0.0,
) -> torch.Tensor:
    """Convert sparse features into a dense volume."""
    feats = sparse_tensor.F if hasattr(sparse_tensor, "F") else sparse_tensor
    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)

    device = feats.device
    coords = coords.to(device=device, dtype=torch.long)

    # clamp coordinates to valid range to avoid out-of-bounds writes
    vol_dim_tensor = torch.as_tensor(vol_dim, device=device, dtype=torch.long)
    vol_dim_tensor = torch.clamp(vol_dim_tensor, min=1)
    coords[:, 0] = coords[:, 0].clamp(0, vol_dim_tensor[0] - 1)
    coords[:, 1] = coords[:, 1].clamp(0, vol_dim_tensor[1] - 1)
    coords[:, 2] = coords[:, 2].clamp(0, vol_dim_tensor[2] - 1)

    dense = torch.full(
        (vol_dim_tensor[0], vol_dim_tensor[1], vol_dim_tensor[2], feats.shape[-1]),
        float(default_val),
        device=device,
        dtype=feats.dtype,
    )
    dense[coords[:, 0], coords[:, 1], coords[:, 2]] = feats
    return dense


def _to_numpy(data: TensorLike) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def construct_sparse_tensor(
    raw_coords: TensorLike,
    feats: TensorLike,
    Bbx_min: TensorLike,
    Bbx_max: TensorLike,
    voxel_size: float = 0.1,
    device: Optional[torch.device] = None,
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize point cloud features into a SparseTensor, preserving gradients from feats.
    """
    if device is None:
        if isinstance(feats, torch.Tensor):
            device = feats.device
        elif isinstance(raw_coords, torch.Tensor):
            device = raw_coords.device
        else:
            device = torch.device("cpu")

    coords_np = _to_numpy(raw_coords)
    feats_tensor = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(feats, dtype=torch.float32)
    
    # Create a leaf node if feats_tensor doesn't have requires_grad or is detached
    # This ensures gradients can flow through the sparse convolution
    if isinstance(feats_tensor, torch.Tensor):
        if not feats_tensor.requires_grad or feats_tensor.grad_fn is None:
            # Create a leaf node by cloning and setting requires_grad
            feats_tensor = feats_tensor.clone().detach().requires_grad_(True)
    feats_tensor = feats_tensor.to(device=device, dtype=torch.float32)

    bbx_max = _to_numpy(Bbx_max).astype(np.float32)
    bbx_min = _to_numpy(Bbx_min).astype(np.float32)
    vol_dim = ((bbx_max - bbx_min) / voxel_size).astype(int).tolist()

    # IMPORTANT:
    # Use the same floating bbx_min offset for voxelization as the grid sampling path.
    # Casting bbx_min to int (truncate-to-zero) silently changes the voxel origin when
    # bbx_min is non-integer (e.g. -19.7 -> -19), which can misalign the dense volume
    # coordinates and the later trilinear interpolation grid coords.
    coords_np = coords_np.astype(np.float32) - bbx_min
    coords_np, indices = sparse_quantize(coords_np, voxel_size, return_index=True)
    coords_th = torch.as_tensor(coords_np, dtype=torch.int32, device=device)
    batch_indices = torch.zeros((coords_th.shape[0], 1), device=device, dtype=torch.int32)
    coords_th = torch.cat((batch_indices, coords_th), dim=1)

    index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
    feats_tensor = feats_tensor[index_tensor]
    sparse_feat = SparseTensor(feats_tensor, coords=coords_th)
    return sparse_feat, torch.as_tensor(vol_dim, device=device, dtype=torch.long), coords_th[:, 1:]


__all__ = [
    "SparseCostRegNet",
    "BasicSparseConvolutionBlock",
    "BasicSparseDeconvolutionBlock",
    "SparseResidualBlock",
    "ConvBnReLU",
    "ConvBnReLU3D",
    "FeatureNet",
    "construct_sparse_tensor",
    "sparse_to_dense_volume",
]
