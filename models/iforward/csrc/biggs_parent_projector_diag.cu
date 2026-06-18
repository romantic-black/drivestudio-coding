#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t softplus_device(scalar_t x) {
    return x > scalar_t(20) ? x : log1p(exp(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_device(scalar_t x, scalar_t lo, scalar_t hi) {
    return fmin(fmax(x, lo), hi);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t top2_area_device(scalar_t x, scalar_t y, scalar_t z) {
    scalar_t a = x;
    scalar_t b = y;
    scalar_t c = z;
    scalar_t lo = fmin(a, fmin(b, c));
    return (a * b * c) / fmax(lo, scalar_t(1.0e-30));
}

template <typename scalar_t>
__device__ __forceinline__ void normalize_quat(
    scalar_t qw,
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t &ow,
    scalar_t &ox,
    scalar_t &oy,
    scalar_t &oz) {
    scalar_t norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + scalar_t(1.0e-8);
    ow = qw / norm;
    ox = qx / norm;
    oy = qy / norm;
    oz = qz / norm;
}

template <typename scalar_t>
__device__ __forceinline__ void child_diag_cov_device(
    const scalar_t *quats,
    const scalar_t sx2,
    const scalar_t sy2,
    const scalar_t sz2,
    int64_t i,
    scalar_t &dx,
    scalar_t &dy,
    scalar_t &dz) {
    scalar_t w, x, y, z;
    normalize_quat(
        quats[i * 4 + 0],
        quats[i * 4 + 1],
        quats[i * 4 + 2],
        quats[i * 4 + 3],
        w,
        x,
        y,
        z);
    scalar_t r00 = scalar_t(1) - scalar_t(2) * (y * y + z * z);
    scalar_t r01 = scalar_t(2) * (x * y - w * z);
    scalar_t r02 = scalar_t(2) * (x * z + w * y);
    scalar_t r10 = scalar_t(2) * (x * y + w * z);
    scalar_t r11 = scalar_t(1) - scalar_t(2) * (x * x + z * z);
    scalar_t r12 = scalar_t(2) * (y * z - w * x);
    scalar_t r20 = scalar_t(2) * (x * z - w * y);
    scalar_t r21 = scalar_t(2) * (y * z + w * x);
    scalar_t r22 = scalar_t(1) - scalar_t(2) * (x * x + y * y);
    dx = r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2;
    dy = r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2;
    dz = r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2;
}

template <typename scalar_t>
__global__ void biggs_parent_project_diag_forward_kernel(
    const scalar_t *__restrict__ means,
    const scalar_t *__restrict__ scales_log,
    const scalar_t *__restrict__ quats,
    const scalar_t *__restrict__ opacity_logit,
    const scalar_t *__restrict__ sh_dc,
    const scalar_t *__restrict__ sh_rest,
    const scalar_t *__restrict__ child_mass,
    const int64_t *__restrict__ child_order,
    const int64_t *__restrict__ parent_start,
    const int64_t *__restrict__ parent_count,
    scalar_t *__restrict__ parent_means,
    scalar_t *__restrict__ parent_scales_log,
    scalar_t *__restrict__ parent_quats,
    scalar_t *__restrict__ parent_opacity_logit,
    scalar_t *__restrict__ parent_sh_dc,
    scalar_t *__restrict__ parent_sh_rest,
    scalar_t *__restrict__ mass_sum_out,
    scalar_t *__restrict__ mass_mean_out,
    scalar_t *__restrict__ weighted_mean_sum_out,
    scalar_t *__restrict__ weighted_second_sum_out,
    scalar_t *__restrict__ tau_area_sum_out,
    scalar_t *__restrict__ weighted_sh_dc_sum_out,
    scalar_t *__restrict__ weighted_sh_rest_sum_out,
    scalar_t *__restrict__ child_cache_mass_out,
    scalar_t *__restrict__ child_cache_tau_area_out,
    scalar_t *__restrict__ child_cache_diag_cov_out,
    int64_t m,
    int64_t sh_bases,
    scalar_t min_scale,
    scalar_t max_scale,
    scalar_t opacity_cap,
    scalar_t opacity_min,
    scalar_t tau_parent_scale,
    scalar_t eps,
    scalar_t min_mass,
    int64_t mass_mode) {
    int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= m) {
        return;
    }

    int64_t start = parent_start[p];
    int64_t count = parent_count[p];
    double W = 0.0;
    double A0 = 0.0, A1 = 0.0, A2 = 0.0;
    double U = 0.0;
    double C0 = 0.0, C1 = 0.0, C2 = 0.0;

    for (int64_t local = 0; local < count; ++local) {
        int64_t i = child_order[start + local];
        scalar_t mx = means[i * 3 + 0];
        scalar_t my = means[i * 3 + 1];
        scalar_t mz = means[i * 3 + 2];
        scalar_t sx = exp(scales_log[i * 3 + 0]);
        scalar_t sy = exp(scales_log[i * 3 + 1]);
        scalar_t sz = exp(scales_log[i * 3 + 2]);
        scalar_t area = top2_area_device(sx, sy, sz);
        scalar_t tau = softplus_device(opacity_logit[i]);
        scalar_t mass = mass_mode == 1 ? child_mass[i] : tau * area;
        mass = fmax(mass, min_mass);
        scalar_t tau_area = tau * area;
        double mass_d = static_cast<double>(mass);
        double tau_area_d = static_cast<double>(tau_area);
        double mx_d = static_cast<double>(mx);
        double my_d = static_cast<double>(my);
        double mz_d = static_cast<double>(mz);

        scalar_t dx, dy, dz;
        child_diag_cov_device(quats, sx * sx, sy * sy, sz * sz, i, dx, dy, dz);

        if (child_cache_mass_out != nullptr) {
            child_cache_mass_out[i] = mass;
            child_cache_tau_area_out[i] = tau_area;
            child_cache_diag_cov_out[i * 3 + 0] = dx;
            child_cache_diag_cov_out[i * 3 + 1] = dy;
            child_cache_diag_cov_out[i * 3 + 2] = dz;
        }

        W += mass_d;
        A0 += mass_d * mx_d;
        A1 += mass_d * my_d;
        A2 += mass_d * mz_d;
        U += tau_area_d;
        C0 += mass_d * static_cast<double>(sh_dc[i * 3 + 0]);
        C1 += mass_d * static_cast<double>(sh_dc[i * 3 + 1]);
        C2 += mass_d * static_cast<double>(sh_dc[i * 3 + 2]);
    }

    double safeW = fmax(W, static_cast<double>(min_mass));
    double mean0 = A0 / safeW;
    double mean1 = A1 / safeW;
    double mean2 = A2 / safeW;
    double B0 = 0.0, B1 = 0.0, B2 = 0.0;
    for (int64_t local = 0; local < count; ++local) {
        int64_t i = child_order[start + local];
        scalar_t mx = means[i * 3 + 0];
        scalar_t my = means[i * 3 + 1];
        scalar_t mz = means[i * 3 + 2];
        scalar_t sx = exp(scales_log[i * 3 + 0]);
        scalar_t sy = exp(scales_log[i * 3 + 1]);
        scalar_t sz = exp(scales_log[i * 3 + 2]);
        scalar_t area = top2_area_device(sx, sy, sz);
        scalar_t tau = softplus_device(opacity_logit[i]);
        scalar_t mass = mass_mode == 1 ? child_mass[i] : tau * area;
        mass = fmax(mass, min_mass);
        scalar_t dx, dy, dz;
        child_diag_cov_device(quats, sx * sx, sy * sy, sz * sz, i, dx, dy, dz);
        double rx = static_cast<double>(mx) - mean0;
        double ry = static_cast<double>(my) - mean1;
        double rz = static_cast<double>(mz) - mean2;
        double mass_d = static_cast<double>(mass);
        B0 += mass_d * (static_cast<double>(dx) + rx * rx);
        B1 += mass_d * (static_cast<double>(dy) + ry * ry);
        B2 += mass_d * (static_cast<double>(dz) + rz * rz);
    }
    double min_var = static_cast<double>(min_scale) * static_cast<double>(min_scale);
    double max_var = static_cast<double>(max_scale) * static_cast<double>(max_scale);
    double eps_d = static_cast<double>(eps);
    double var0 = fmin(fmax(B0 / safeW + eps_d, min_var), max_var);
    double var1 = fmin(fmax(B1 / safeW + eps_d, min_var), max_var);
    double var2 = fmin(fmax(B2 / safeW + eps_d, min_var), max_var);
    double ps0 = fmin(fmax(sqrt(var0), static_cast<double>(min_scale)), static_cast<double>(max_scale));
    double ps1 = fmin(fmax(sqrt(var1), static_cast<double>(min_scale)), static_cast<double>(max_scale));
    double ps2 = fmin(fmax(sqrt(var2), static_cast<double>(min_scale)), static_cast<double>(max_scale));
    double parent_area = fmax(top2_area_device(ps0, ps1, ps2), eps_d);
    double tau_parent = static_cast<double>(tau_parent_scale) * U / (parent_area + eps_d);
    double opacity = static_cast<double>(opacity_cap) * (1.0 - exp(-tau_parent));
    opacity = fmin(fmax(opacity, static_cast<double>(opacity_min)), static_cast<double>(opacity_cap) - eps_d);
    double logit = log(opacity / fmax(1.0 - opacity, eps_d));

    parent_means[p * 3 + 0] = count > 0 ? static_cast<scalar_t>(mean0) : scalar_t(0);
    parent_means[p * 3 + 1] = count > 0 ? static_cast<scalar_t>(mean1) : scalar_t(0);
    parent_means[p * 3 + 2] = count > 0 ? static_cast<scalar_t>(mean2) : scalar_t(0);
    parent_scales_log[p * 3 + 0] = static_cast<scalar_t>(log(fmax(ps0, static_cast<double>(min_scale))));
    parent_scales_log[p * 3 + 1] = static_cast<scalar_t>(log(fmax(ps1, static_cast<double>(min_scale))));
    parent_scales_log[p * 3 + 2] = static_cast<scalar_t>(log(fmax(ps2, static_cast<double>(min_scale))));
    parent_quats[p * 4 + 0] = scalar_t(1);
    parent_quats[p * 4 + 1] = scalar_t(0);
    parent_quats[p * 4 + 2] = scalar_t(0);
    parent_quats[p * 4 + 3] = scalar_t(0);
    parent_opacity_logit[p] = static_cast<scalar_t>(logit);
    parent_sh_dc[p * 3 + 0] = count > 0 ? static_cast<scalar_t>(C0 / safeW) : scalar_t(0);
    parent_sh_dc[p * 3 + 1] = count > 0 ? static_cast<scalar_t>(C1 / safeW) : scalar_t(0);
    parent_sh_dc[p * 3 + 2] = count > 0 ? static_cast<scalar_t>(C2 / safeW) : scalar_t(0);
    mass_sum_out[p] = static_cast<scalar_t>(W);
    mass_mean_out[p] = static_cast<scalar_t>(W / fmax(static_cast<double>(count), 1.0));
    if (weighted_mean_sum_out != nullptr) {
        weighted_mean_sum_out[p * 3 + 0] = static_cast<scalar_t>(A0);
        weighted_mean_sum_out[p * 3 + 1] = static_cast<scalar_t>(A1);
        weighted_mean_sum_out[p * 3 + 2] = static_cast<scalar_t>(A2);
        weighted_second_sum_out[p * 3 + 0] = static_cast<scalar_t>(B0);
        weighted_second_sum_out[p * 3 + 1] = static_cast<scalar_t>(B1);
        weighted_second_sum_out[p * 3 + 2] = static_cast<scalar_t>(B2);
        tau_area_sum_out[p] = static_cast<scalar_t>(U);
        weighted_sh_dc_sum_out[p * 3 + 0] = static_cast<scalar_t>(C0);
        weighted_sh_dc_sum_out[p * 3 + 1] = static_cast<scalar_t>(C1);
        weighted_sh_dc_sum_out[p * 3 + 2] = static_cast<scalar_t>(C2);
    }

    for (int64_t b = 0; b < sh_bases; ++b) {
        double r0 = 0.0, r1 = 0.0, r2 = 0.0;
        for (int64_t local = 0; local < count; ++local) {
            int64_t i = child_order[start + local];
            scalar_t sx = exp(scales_log[i * 3 + 0]);
            scalar_t sy = exp(scales_log[i * 3 + 1]);
            scalar_t sz = exp(scales_log[i * 3 + 2]);
            scalar_t area = top2_area_device(sx, sy, sz);
            scalar_t tau = softplus_device(opacity_logit[i]);
            scalar_t mass = mass_mode == 1 ? child_mass[i] : tau * area;
            mass = fmax(mass, min_mass);
            double mass_d = static_cast<double>(mass);
            int64_t off = (i * sh_bases + b) * 3;
            r0 += mass_d * static_cast<double>(sh_rest[off + 0]);
            r1 += mass_d * static_cast<double>(sh_rest[off + 1]);
            r2 += mass_d * static_cast<double>(sh_rest[off + 2]);
        }
        int64_t poff = (p * sh_bases + b) * 3;
        parent_sh_rest[poff + 0] = count > 0 ? static_cast<scalar_t>(r0 / safeW) : scalar_t(0);
        parent_sh_rest[poff + 1] = count > 0 ? static_cast<scalar_t>(r1 / safeW) : scalar_t(0);
        parent_sh_rest[poff + 2] = count > 0 ? static_cast<scalar_t>(r2 / safeW) : scalar_t(0);
        if (weighted_sh_rest_sum_out != nullptr) {
            weighted_sh_rest_sum_out[poff + 0] = static_cast<scalar_t>(r0);
            weighted_sh_rest_sum_out[poff + 1] = static_cast<scalar_t>(r1);
            weighted_sh_rest_sum_out[poff + 2] = static_cast<scalar_t>(r2);
        }
    }
}

}  // namespace

std::vector<torch::Tensor> biggs_parent_project_diag_forward_cuda(
    torch::Tensor means,
    torch::Tensor scales_log,
    torch::Tensor quats,
    torch::Tensor opacity_logit,
    torch::Tensor sh_dc,
    torch::Tensor sh_rest,
    torch::Tensor child_mass,
    torch::Tensor child_order,
    torch::Tensor parent_start,
    torch::Tensor parent_count,
    double min_scale,
    double max_scale,
    double opacity_cap,
    double opacity_min,
    double tau_parent_scale,
    double eps,
    double min_mass,
    int64_t mass_mode) {
    const auto m = parent_count.numel();
    const auto sh_bases = sh_rest.size(1);
    auto parent_means = torch::empty({m, 3}, means.options());
    auto parent_scales_log = torch::empty({m, 3}, means.options());
    auto parent_quats = torch::empty({m, 4}, means.options());
    auto parent_opacity_logit = torch::empty({m, 1}, means.options());
    auto parent_sh_dc = torch::empty({m, 3}, means.options());
    auto parent_sh_rest = torch::empty({m, sh_bases, 3}, means.options());
    auto mass_sum = torch::empty({m}, means.options());
    auto mass_mean = torch::empty({m}, means.options());

    if (m == 0) {
        return {parent_means, parent_scales_log, parent_quats, parent_opacity_logit, parent_sh_dc, parent_sh_rest, mass_sum, mass_mean};
    }

    const int threads = 128;
    const int blocks = static_cast<int>((m + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "biggs_parent_project_diag_forward_cuda", [&] {
        biggs_parent_project_diag_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            means.data_ptr<scalar_t>(),
            scales_log.data_ptr<scalar_t>(),
            quats.data_ptr<scalar_t>(),
            opacity_logit.data_ptr<scalar_t>(),
            sh_dc.data_ptr<scalar_t>(),
            sh_rest.data_ptr<scalar_t>(),
            child_mass.data_ptr<scalar_t>(),
            child_order.data_ptr<int64_t>(),
            parent_start.data_ptr<int64_t>(),
            parent_count.data_ptr<int64_t>(),
            parent_means.data_ptr<scalar_t>(),
            parent_scales_log.data_ptr<scalar_t>(),
            parent_quats.data_ptr<scalar_t>(),
            parent_opacity_logit.data_ptr<scalar_t>(),
            parent_sh_dc.data_ptr<scalar_t>(),
            parent_sh_rest.data_ptr<scalar_t>(),
            mass_sum.data_ptr<scalar_t>(),
            mass_mean.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            m,
            sh_bases,
            static_cast<scalar_t>(min_scale),
            static_cast<scalar_t>(max_scale),
            static_cast<scalar_t>(opacity_cap),
            static_cast<scalar_t>(opacity_min),
            static_cast<scalar_t>(tau_parent_scale),
            static_cast<scalar_t>(eps),
            static_cast<scalar_t>(min_mass),
            mass_mode);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {parent_means, parent_scales_log, parent_quats, parent_opacity_logit, parent_sh_dc, parent_sh_rest, mass_sum, mass_mean};
}

std::vector<torch::Tensor> biggs_parent_project_diag_forward_with_stats_cuda(
    torch::Tensor means,
    torch::Tensor scales_log,
    torch::Tensor quats,
    torch::Tensor opacity_logit,
    torch::Tensor sh_dc,
    torch::Tensor sh_rest,
    torch::Tensor child_mass,
    torch::Tensor child_order,
    torch::Tensor parent_start,
    torch::Tensor parent_count,
    double min_scale,
    double max_scale,
    double opacity_cap,
    double opacity_min,
    double tau_parent_scale,
    double eps,
    double min_mass,
    int64_t mass_mode) {
    const auto n = means.size(0);
    const auto m = parent_count.numel();
    const auto sh_bases = sh_rest.size(1);
    auto parent_means = torch::empty({m, 3}, means.options());
    auto parent_scales_log = torch::empty({m, 3}, means.options());
    auto parent_quats = torch::empty({m, 4}, means.options());
    auto parent_opacity_logit = torch::empty({m, 1}, means.options());
    auto parent_sh_dc = torch::empty({m, 3}, means.options());
    auto parent_sh_rest = torch::empty({m, sh_bases, 3}, means.options());
    auto mass_sum = torch::empty({m}, means.options());
    auto mass_mean = torch::empty({m}, means.options());
    auto weighted_mean_sum = torch::empty({m, 3}, means.options());
    auto weighted_second_sum = torch::empty({m, 3}, means.options());
    auto tau_area_sum = torch::empty({m}, means.options());
    auto weighted_sh_dc_sum = torch::empty({m, 3}, means.options());
    auto weighted_sh_rest_sum = torch::empty({m, sh_bases, 3}, means.options());
    auto child_cache_mass = torch::zeros({n}, means.options());
    auto child_cache_tau_area = torch::zeros({n}, means.options());
    auto child_cache_diag_cov = torch::zeros({n, 3}, means.options());

    if (m == 0) {
        return {
            parent_means,
            parent_scales_log,
            parent_quats,
            parent_opacity_logit,
            parent_sh_dc,
            parent_sh_rest,
            mass_sum,
            mass_mean,
            weighted_mean_sum,
            weighted_second_sum,
            tau_area_sum,
            weighted_sh_dc_sum,
            weighted_sh_rest_sum,
            child_cache_mass,
            child_cache_tau_area,
            child_cache_diag_cov,
        };
    }

    const int threads = 128;
    const int blocks = static_cast<int>((m + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "biggs_parent_project_diag_forward_with_stats_cuda", [&] {
        biggs_parent_project_diag_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            means.data_ptr<scalar_t>(),
            scales_log.data_ptr<scalar_t>(),
            quats.data_ptr<scalar_t>(),
            opacity_logit.data_ptr<scalar_t>(),
            sh_dc.data_ptr<scalar_t>(),
            sh_rest.data_ptr<scalar_t>(),
            child_mass.data_ptr<scalar_t>(),
            child_order.data_ptr<int64_t>(),
            parent_start.data_ptr<int64_t>(),
            parent_count.data_ptr<int64_t>(),
            parent_means.data_ptr<scalar_t>(),
            parent_scales_log.data_ptr<scalar_t>(),
            parent_quats.data_ptr<scalar_t>(),
            parent_opacity_logit.data_ptr<scalar_t>(),
            parent_sh_dc.data_ptr<scalar_t>(),
            parent_sh_rest.data_ptr<scalar_t>(),
            mass_sum.data_ptr<scalar_t>(),
            mass_mean.data_ptr<scalar_t>(),
            weighted_mean_sum.data_ptr<scalar_t>(),
            weighted_second_sum.data_ptr<scalar_t>(),
            tau_area_sum.data_ptr<scalar_t>(),
            weighted_sh_dc_sum.data_ptr<scalar_t>(),
            weighted_sh_rest_sum.data_ptr<scalar_t>(),
            child_cache_mass.data_ptr<scalar_t>(),
            child_cache_tau_area.data_ptr<scalar_t>(),
            child_cache_diag_cov.data_ptr<scalar_t>(),
            m,
            sh_bases,
            static_cast<scalar_t>(min_scale),
            static_cast<scalar_t>(max_scale),
            static_cast<scalar_t>(opacity_cap),
            static_cast<scalar_t>(opacity_min),
            static_cast<scalar_t>(tau_parent_scale),
            static_cast<scalar_t>(eps),
            static_cast<scalar_t>(min_mass),
            mass_mode);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        parent_means,
        parent_scales_log,
        parent_quats,
        parent_opacity_logit,
        parent_sh_dc,
        parent_sh_rest,
        mass_sum,
        mass_mean,
        weighted_mean_sum,
        weighted_second_sum,
        tau_area_sum,
        weighted_sh_dc_sum,
        weighted_sh_rest_sum,
        child_cache_mass,
        child_cache_tau_area,
        child_cache_diag_cov,
    };
}
