#include <torch/extension.h>

#include <vector>

torch::Tensor grld_decode_forward_cuda(
    torch::Tensor base,
    torch::Tensor detail,
    torch::Tensor gate,
    torch::Tensor coeff,
    torch::Tensor child_to_parent,
    torch::Tensor branch_scale);

std::vector<torch::Tensor> grld_decode_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor base,
    torch::Tensor detail,
    torch::Tensor gate,
    torch::Tensor coeff,
    torch::Tensor child_to_parent,
    torch::Tensor child_order,
    torch::Tensor parent_start,
    torch::Tensor parent_count,
    torch::Tensor branch_scale);

torch::Tensor grld_decode_forward(
    torch::Tensor base,
    torch::Tensor detail,
    torch::Tensor gate,
    torch::Tensor coeff,
    torch::Tensor child_to_parent,
    torch::Tensor branch_scale) {
    TORCH_CHECK(base.is_cuda(), "grld_decode_forward requires CUDA tensors");
    TORCH_CHECK(detail.is_cuda() && gate.is_cuda() && coeff.is_cuda(), "GRLD inputs must be CUDA tensors");
    TORCH_CHECK(child_to_parent.is_cuda() && branch_scale.is_cuda(), "GRLD index/scale tensors must be CUDA tensors");
    return grld_decode_forward_cuda(base, detail, gate, coeff, child_to_parent, branch_scale);
}

std::vector<torch::Tensor> grld_decode_backward(
    torch::Tensor grad_out,
    torch::Tensor base,
    torch::Tensor detail,
    torch::Tensor gate,
    torch::Tensor coeff,
    torch::Tensor child_to_parent,
    torch::Tensor child_order,
    torch::Tensor parent_start,
    torch::Tensor parent_count,
    torch::Tensor branch_scale) {
    TORCH_CHECK(grad_out.is_cuda(), "grld_decode_backward requires CUDA tensors");
    TORCH_CHECK(base.is_cuda() && detail.is_cuda() && gate.is_cuda() && coeff.is_cuda(), "GRLD saved inputs must be CUDA tensors");
    TORCH_CHECK(child_to_parent.is_cuda() && child_order.is_cuda() && parent_start.is_cuda() && parent_count.is_cuda(), "GRLD index tensors must be CUDA tensors");
    TORCH_CHECK(branch_scale.is_cuda(), "GRLD branch scale must be a CUDA tensor");
    return grld_decode_backward_cuda(grad_out, base, detail, gate, coeff, child_to_parent, child_order, parent_start, parent_count, branch_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grld_decode_forward", &grld_decode_forward, "GRLD fused decode forward (CUDA)");
    m.def("grld_decode_backward", &grld_decode_backward, "GRLD fused decode backward (CUDA)");
}
