"""
Sparse convolution components extracted from EVolSplat.
"""
from torch import nn
import torch
from torchsparse import nn as spnn
import numpy as np
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.tensor import SparseTensor


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                stride=stride,
                transposed=True
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class SparseCostRegNet(nn.Module):
    """Sparse cost regularization network for 3D volume feature extraction."""

    def __init__(self, d_in, d_out=8):
        super(SparseCostRegNet, self).__init__()
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

    def forward(self, x):
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


def sparse_to_dense_volume(sparse_tensor, coords, vol_dim, default_val=0):
    """Convert sparse tensor to dense volume.
    
    Args:
        sparse_tensor: Sparse tensor features [N, C]
        coords: Coordinates [N, 3]
        vol_dim: Volume dimensions [H, W, D]
        default_val: Default value for empty voxels
        
    Returns:
        Dense volume [H, W, D, C]
    """
    c = sparse_tensor.shape[-1]
    coords = coords.to(torch.int64)
    # Clamp the coords to prevent data overflow
    coords[:, 0] = coords[:, 0].clamp(0, vol_dim[0] - 1)
    coords[:, 1] = coords[:, 1].clamp(0, vol_dim[1] - 1)
    coords[:, 2] = coords[:, 2].clamp(0, vol_dim[2] - 1)
    device = sparse_tensor.device
    dense = torch.full(
        [vol_dim[0], vol_dim[1], vol_dim[2], c],
        float(default_val),
        device=device
    )
    dense[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_tensor
    return dense


def construct_sparse_tensor(
    raw_coords,
    feats,
    Bbx_min: torch.Tensor,
    Bbx_max: torch.Tensor,
    voxel_size=0.1
):
    """Construct sparse tensor from raw coordinates and features.
    
    Args:
        raw_coords: Raw 3D coordinates [N, 3]
        feats: Features [N, C]
        Bbx_min: Bounding box minimum [3]
        Bbx_max: Bounding box maximum [3]
        voxel_size: Voxel size for quantization
        
    Returns:
        sparse_feat: SparseTensor
        vol_dim: Volume dimensions [H, W, D]
        valid_coords: Valid coordinates [N, 3]
    """
    X_MIN, X_MAX = Bbx_min[0], Bbx_max[0]
    Y_MIN, Y_MAX = Bbx_min[1], Bbx_max[1]
    Z_MIN, Z_MAX = Bbx_min[2], Bbx_max[2]

    if isinstance(raw_coords, torch.Tensor):
        raw_coords = raw_coords.cpu().numpy()
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()

    bbx_max = np.array([X_MAX, Y_MAX, Z_MAX])
    bbx_min = np.array([X_MIN, Y_MIN, Z_MIN])
    vol_dim = (bbx_max - bbx_min) / voxel_size
    vol_dim = vol_dim.astype(int).tolist()

    raw_coords -= np.array([X_MIN, Y_MIN, Z_MIN]).astype(int)
    coords, indices = sparse_quantize(
        raw_coords, voxel_size, return_index=True
    )  # Voxelize the points to discrete formation
    coords = torch.tensor(coords, dtype=torch.int).cuda()
    zeros = torch.zeros(coords.shape[0], 1).cuda()
    # Note: [B, X, Y, Z] in Torch sparse 2.1
    coords = torch.cat((zeros, coords), dim=1).to(torch.int32)

    feats = torch.tensor(feats[indices], dtype=torch.float).cuda()
    sparse_feat = SparseTensor(feats, coords=coords)
    return sparse_feat, vol_dim, coords[:, 1:]

