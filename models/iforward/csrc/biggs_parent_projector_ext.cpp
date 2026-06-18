#include <torch/extension.h>

#include <vector>

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
    int64_t mass_mode);

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
    int64_t mass_mode);

std::vector<torch::Tensor> biggs_parent_project_diag_forward(
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
    TORCH_CHECK(means.is_cuda(), "biggs_parent_project_diag_forward requires CUDA tensors");
    TORCH_CHECK(scales_log.is_cuda() && quats.is_cuda() && opacity_logit.is_cuda(), "all child params must be CUDA tensors");
    TORCH_CHECK(sh_dc.is_cuda() && sh_rest.is_cuda(), "SH tensors must be CUDA tensors");
    TORCH_CHECK(child_mass.is_cuda() && child_order.is_cuda() && parent_start.is_cuda() && parent_count.is_cuda(), "assignment tensors must be CUDA tensors");
    return biggs_parent_project_diag_forward_cuda(
        means,
        scales_log,
        quats,
        opacity_logit,
        sh_dc,
        sh_rest,
        child_mass,
        child_order,
        parent_start,
        parent_count,
        min_scale,
        max_scale,
        opacity_cap,
        opacity_min,
        tau_parent_scale,
        eps,
        min_mass,
        mass_mode);
}

std::vector<torch::Tensor> biggs_parent_project_diag_forward_with_stats(
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
    TORCH_CHECK(means.is_cuda(), "biggs_parent_project_diag_forward_with_stats requires CUDA tensors");
    TORCH_CHECK(scales_log.is_cuda() && quats.is_cuda() && opacity_logit.is_cuda(), "all child params must be CUDA tensors");
    TORCH_CHECK(sh_dc.is_cuda() && sh_rest.is_cuda(), "SH tensors must be CUDA tensors");
    TORCH_CHECK(child_mass.is_cuda() && child_order.is_cuda() && parent_start.is_cuda() && parent_count.is_cuda(), "assignment tensors must be CUDA tensors");
    return biggs_parent_project_diag_forward_with_stats_cuda(
        means,
        scales_log,
        quats,
        opacity_logit,
        sh_dc,
        sh_rest,
        child_mass,
        child_order,
        parent_start,
        parent_count,
        min_scale,
        max_scale,
        opacity_cap,
        opacity_min,
        tau_parent_scale,
        eps,
        min_mass,
        mass_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("biggs_parent_project_diag_forward", &biggs_parent_project_diag_forward, "BigGS diagonal parent projection forward (CUDA)");
    m.def("biggs_parent_project_diag_forward_with_stats", &biggs_parent_project_diag_forward_with_stats, "BigGS diagonal parent projection forward with runtime stats (CUDA)");
}
