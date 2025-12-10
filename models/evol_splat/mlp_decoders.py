"""
MLP decoders extracted from EVolSplat for generating Gaussian parameters.
"""
import torch
import torch.nn as nn

# Try to import MLP from nerfstudio, fallback to simple implementation
try:
    from nerfstudio.field_components.mlp import MLP
except ImportError:
    # Fallback: simple MLP implementation
    class MLP(nn.Module):
        """Simple MLP implementation as fallback."""
        
        def __init__(
            self,
            in_dim: int,
            num_layers: int,
            layer_width: int,
            out_dim: int,
            activation: nn.Module = nn.ReLU(),
            out_activation: nn.Module = None,
            implementation: str = "torch",
        ):
            super().__init__()
            layers = []
            if num_layers == 1:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                for i in range(num_layers - 1):
                    if i == 0:
                        layers.append(nn.Linear(in_dim, layer_width))
                    else:
                        layers.append(nn.Linear(layer_width, layer_width))
                layers.append(nn.Linear(layer_width, out_dim))
            
            self.layers = nn.ModuleList(layers)
            self.activation = activation
            self.out_activation = out_activation
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.activation is not None and i < len(self.layers) - 1:
                    x = self.activation(x)
            if self.out_activation is not None:
                x = self.out_activation(x)
            return x


def create_gaussion_decoder(feature_dim_in, feature_dim_out, sh_degree=1):
    """Create the Gaussian appearance decoder MLP.
    
    Args:
        feature_dim_in: Input feature dimension
        feature_dim_out: Output feature dimension (3 * num_sh_bases)
        sh_degree: Spherical harmonics degree
        
    Returns:
        MLP decoder
    """
    return MLP(
        in_dim=feature_dim_in + 4,  # +4 for ob_dist and ob_view
        num_layers=3,
        layer_width=128,
        out_dim=feature_dim_out,
        activation=nn.ReLU(),
        out_activation=None,
        implementation="torch",
    )


def create_mlp_conv(sparse_conv_outdim):
    """Create MLP for scale and rotation prediction.
    
    Args:
        sparse_conv_outdim: Output dimension of sparse convolution
        
    Returns:
        MLP decoder
    """
    return MLP(
        in_dim=sparse_conv_outdim + 4,  # +4 for ob_dist and ob_view
        num_layers=2,
        layer_width=64,
        out_dim=3 + 4,  # 3 for scale, 4 for quaternion
        activation=nn.Tanh(),
        out_activation=None,
        implementation="torch",
    )


def create_mlp_opacity(sparse_conv_outdim):
    """Create MLP for opacity prediction.
    
    Args:
        sparse_conv_outdim: Output dimension of sparse convolution
        
    Returns:
        MLP decoder
    """
    return MLP(
        in_dim=sparse_conv_outdim + 4,  # +4 for ob_dist and ob_view
        num_layers=2,
        layer_width=64,
        out_dim=1,
        activation=nn.ReLU(),
        out_activation=None,
        implementation="torch",
    )


def create_mlp_offset(sparse_conv_outdim):
    """Create MLP for 3D offset prediction.
    
    Args:
        sparse_conv_outdim: Output dimension of sparse convolution
        
    Returns:
        MLP decoder
    """
    return MLP(
        in_dim=sparse_conv_outdim,
        num_layers=2,
        layer_width=64,
        out_dim=3,
        activation=nn.ReLU(),
        out_activation=nn.Tanh(),
        implementation="torch",
    )

