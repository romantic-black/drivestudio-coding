from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from gsplat.cuda._torch_impl import _spherical_harmonics as torch_spherical_harmonics
from gsplat.cuda._wrapper import spherical_harmonics as cuda_spherical_harmonics

from models.feature_extractors.alpha_t_extractor import _get_viewmat


@dataclass
class UncertaintyRenderBundle:
    rgb: torch.Tensor
    alpha: torch.Tensor
    # `variance`/`logvar` remain the total diagnostic moment for compatibility.
    variance: torch.Tensor
    logvar: torch.Tensor
    aleatoric_variance: torch.Tensor
    disagreement_variance: torch.Tensor
    within_variance: torch.Tensor
    background_variance: torch.Tensor
    loss_variance: torch.Tensor
    loss_logvar: torch.Tensor
    total_variance: torch.Tensor
    total_logvar: torch.Tensor

    def select_view(self, index: int) -> "UncertaintyRenderBundle":
        return UncertaintyRenderBundle(
            rgb=self.rgb[int(index)],
            alpha=self.alpha[int(index)],
            variance=self.variance[int(index)],
            logvar=self.logvar[int(index)],
            aleatoric_variance=self.aleatoric_variance[int(index)],
            disagreement_variance=self.disagreement_variance[int(index)],
            within_variance=self.within_variance[int(index)],
            background_variance=self.background_variance[int(index)],
            loss_variance=self.loss_variance[int(index)],
            loss_logvar=self.loss_logvar[int(index)],
            total_variance=self.total_variance[int(index)],
            total_logvar=self.total_logvar[int(index)],
        )


@dataclass
class UncertaintyImagePack:
    image_ref: Tuple[int, int]
    role: str
    sigma: torch.Tensor
    variance: torch.Tensor
    aleatoric_variance: torch.Tensor
    disagreement_variance: torch.Tensor
    alpha: torch.Tensor
    within_variance: Optional[torch.Tensor] = None
    background_variance: Optional[torch.Tensor] = None
    total_variance: Optional[torch.Tensor] = None

    def detached_cpu(self) -> "UncertaintyImagePack":
        return UncertaintyImagePack(
            image_ref=(int(self.image_ref[0]), int(self.image_ref[1])),
            role=str(self.role),
            sigma=self.sigma.detach().float().cpu(),
            variance=self.variance.detach().float().cpu(),
            aleatoric_variance=self.aleatoric_variance.detach().float().cpu(),
            disagreement_variance=self.disagreement_variance.detach().float().cpu(),
            alpha=self.alpha.detach().float().cpu(),
            within_variance=(
                self.aleatoric_variance.detach().float().cpu()
                if self.within_variance is None
                else self.within_variance.detach().float().cpu()
            ),
            background_variance=(
                torch.zeros_like(self.variance).detach().float().cpu()
                if self.background_variance is None
                else self.background_variance.detach().float().cpu()
            ),
            total_variance=(
                self.variance.detach().float().cpu()
                if self.total_variance is None
                else self.total_variance.detach().float().cpu()
            ),
        )


def camera_matrices_for_targets(
    targets: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    viewmats = []
    intrinsics = []
    for target in targets:
        view = target["view"]
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmats.append(_get_viewmat(c2w.to(device=device, dtype=dtype)))
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)
        intrinsics.append(k_mat.to(device=device, dtype=dtype))
    return torch.cat(viewmats, dim=0), torch.cat(intrinsics, dim=0)


def _as_batched_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return rgb.unsqueeze(0) if rgb.dim() == 3 else rgb


def _as_batched_alpha(alpha: torch.Tensor, *, camera_count: int) -> torch.Tensor:
    out = alpha
    if out.dim() == 2:
        out = out.unsqueeze(0)
    if out.dim() == 4 and int(out.shape[-1]) == 1:
        out = out.squeeze(-1)
    if out.dim() == 3 and int(out.shape[0]) != int(camera_count) and int(out.shape[-1]) == 1:
        out = out.squeeze(-1).unsqueeze(0)
    return out


def evaluate_spherical_harmonics(
    degree: int,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate SH without requiring a CUDA kernel for CPU reference paths."""
    if directions.is_cuda:
        return cuda_spherical_harmonics(int(degree), directions, coefficients)
    return torch_spherical_harmonics(int(degree), directions, coefficients)


def render_detached_uncertainty_moments(
    *,
    rasterizer: Any,
    render_params: Dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sh_degree: int,
    background_sigma: float = 0.10,
    background_sigma_for_loss: float = 0.10,
    variance_floor: float = 1.0e-4,
    variance_max: float = 0.25,
    variance_mode: str = "total_variance_scalar",
    detach_first_pass_alpha: bool = False,
) -> UncertaintyRenderBundle:
    required = (
        "means_r",
        "quats_r",
        "scales_r",
        "opacities_r",
        "colors_r",
        "appearance_logvar_r",
    )
    missing = [key for key in required if key not in render_params]
    if missing:
        raise ValueError(f"Uncertainty renderer missing render params: {missing}")

    means = render_params["means_r"].detach().float()
    quats = render_params["quats_r"].detach().float()
    scales = render_params["scales_r"].detach().float()
    opacities = render_params["opacities_r"].detach().float()
    sh_coeffs = render_params["colors_r"].detach().float()
    appearance_logvar = render_params["appearance_logvar_r"].float()
    if tuple(appearance_logvar.shape) != (int(means.shape[0]), 1):
        raise ValueError(
            "appearance_logvar_r must be [N,1], got "
            f"{tuple(appearance_logvar.shape)} for N={int(means.shape[0])}"
        )
    viewmats = viewmats.detach().to(device=means.device, dtype=torch.float32)
    Ks = Ks.detach().to(device=means.device, dtype=torch.float32)
    camera_positions = torch.linalg.inv(viewmats)[..., :3, 3]
    dirs = means.unsqueeze(0) - camera_positions.unsqueeze(1)
    coeffs_by_view = sh_coeffs.unsqueeze(0).expand(int(viewmats.shape[0]), -1, -1, -1)
    rgb_per_gaussian = evaluate_spherical_harmonics(int(sh_degree), dirs, coeffs_by_view)
    rgb_per_gaussian = torch.clamp_min(rgb_per_gaussian + 0.5, 0.0).detach()

    variance_per_gaussian = appearance_logvar.exp()
    variance_by_view = variance_per_gaussian.unsqueeze(0).expand(int(viewmats.shape[0]), -1, -1)
    color_energy = rgb_per_gaussian.square().mean(dim=-1, keepdim=True)
    features = torch.cat([color_energy + variance_by_view, variance_by_view], dim=-1)
    rendered, alpha_second, _ = rasterizer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=viewmats,
        Ks=Ks,
        width=int(width),
        height=int(height),
        tile_size=16,
        packed=False,
        near_plane=0.01,
        far_plane=1.0e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
        channel_chunk=32,
    )
    if not torch.isfinite(rendered).all() or not torch.isfinite(alpha_second).all():
        raise RuntimeError("Detached uncertainty raster produced NaN/Inf")

    mode = str(variance_mode).lower()
    if mode not in {"total_variance_scalar", "aleatoric_only"}:
        raise ValueError(f"Unsupported uncertainty variance_mode={variance_mode!r}")
    rgb_b = _as_batched_rgb(rgb).float()
    alpha_b = _as_batched_alpha(alpha, camera_count=int(rendered.shape[0])).float().clamp(0.0, 1.0)
    if bool(detach_first_pass_alpha) or mode == "aleatoric_only":
        alpha_b = alpha_b.detach()
    if int(rgb_b.shape[0]) != int(rendered.shape[0]) or int(alpha_b.shape[0]) != int(rendered.shape[0]):
        raise ValueError(
            f"RGB/alpha/uncertainty camera count mismatch: {rgb_b.shape[0]}, "
            f"{alpha_b.shape[0]}, {rendered.shape[0]}"
        )
    moment2 = rendered[..., 0]
    within = rendered[..., 1]
    rgb_energy = rgb_b.detach().square().mean(dim=-1)
    disagreement = (moment2 - within - rgb_energy).clamp_min(0.0)
    background = (1.0 - alpha_b) * float(background_sigma) ** 2
    aleatoric = within + background
    total_variance = (aleatoric + disagreement).clamp(
        min=float(variance_floor), max=float(variance_max)
    )
    if mode == "aleatoric_only":
        loss_raw = within + (1.0 - alpha_b) * float(background_sigma_for_loss) ** 2
        loss_variance = loss_raw.clamp(min=float(variance_floor), max=float(variance_max))
    else:
        loss_variance = total_variance
    total_logvar = total_variance.log()
    loss_logvar = loss_variance.log()
    return UncertaintyRenderBundle(
        rgb=rgb_b,
        alpha=alpha_b,
        variance=total_variance,
        logvar=total_logvar,
        aleatoric_variance=aleatoric,
        disagreement_variance=disagreement,
        within_variance=within,
        background_variance=background,
        loss_variance=loss_variance,
        loss_logvar=loss_logvar,
        total_variance=total_variance,
        total_logvar=total_logvar,
    )


__all__ = [
    "UncertaintyImagePack",
    "UncertaintyRenderBundle",
    "camera_matrices_for_targets",
    "evaluate_spherical_harmonics",
    "render_detached_uncertainty_moments",
]
