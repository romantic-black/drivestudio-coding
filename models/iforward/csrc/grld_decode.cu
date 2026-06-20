#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void grld_decode_forward_kernel(
    const scalar_t *__restrict__ base,
    const scalar_t *__restrict__ detail,
    const scalar_t *__restrict__ gate,
    const scalar_t *__restrict__ coeff,
    const int64_t *__restrict__ child_to_parent,
    const scalar_t *__restrict__ branch_scale,
    scalar_t *__restrict__ out,
    int64_t n,
    int64_t r,
    int64_t e) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * e;
    if (idx >= total) {
        return;
    }
    int64_t child = idx / e;
    int64_t dim = idx - child * e;
    int64_t p = child_to_parent[child];
    double value = static_cast<double>(base[p * e + dim]);
    double scale = static_cast<double>(branch_scale[0]);
    double residual = 0.0;
    for (int64_t rr = 0; rr < r; ++rr) {
        double c = static_cast<double>(coeff[child * r + rr]);
        double g = static_cast<double>(gate[p * r + rr]);
        double d = static_cast<double>(detail[(p * r + rr) * e + dim]);
        residual += c * g * d;
    }
    out[idx] = static_cast<scalar_t>(value + scale * residual);
}

template <typename scalar_t>
__global__ void grld_grad_coeff_kernel(
    const scalar_t *__restrict__ grad_out,
    const scalar_t *__restrict__ detail,
    const scalar_t *__restrict__ gate,
    const int64_t *__restrict__ child_to_parent,
    const scalar_t *__restrict__ branch_scale,
    scalar_t *__restrict__ grad_coeff,
    int64_t n,
    int64_t r,
    int64_t e) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * r;
    if (idx >= total) {
        return;
    }
    int64_t child = idx / r;
    int64_t rr = idx - child * r;
    int64_t p = child_to_parent[child];
    double scale = static_cast<double>(branch_scale[0]);
    double acc = 0.0;
    double g = static_cast<double>(gate[p * r + rr]);
    for (int64_t dim = 0; dim < e; ++dim) {
        double go = static_cast<double>(grad_out[child * e + dim]);
        double d = static_cast<double>(detail[(p * r + rr) * e + dim]);
        acc += go * g * d;
    }
    grad_coeff[idx] = static_cast<scalar_t>(scale * acc);
}

template <typename scalar_t>
__global__ void grld_parent_backward_kernel(
    const scalar_t *__restrict__ grad_out,
    const scalar_t *__restrict__ detail,
    const scalar_t *__restrict__ gate,
    const scalar_t *__restrict__ coeff,
    const int64_t *__restrict__ child_order,
    const int64_t *__restrict__ parent_start,
    const int64_t *__restrict__ parent_count,
    const scalar_t *__restrict__ branch_scale,
    scalar_t *__restrict__ grad_base,
    scalar_t *__restrict__ grad_detail,
    scalar_t *__restrict__ grad_gate,
    scalar_t *__restrict__ grad_branch_scale,
    int64_t m,
    int64_t r,
    int64_t e) {
    int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= m) {
        return;
    }
    int64_t start = parent_start[p];
    int64_t count = parent_count[p];
    double scale = static_cast<double>(branch_scale[0]);
    double scale_acc = 0.0;
    for (int64_t dim = 0; dim < e; ++dim) {
        double base_acc = 0.0;
        for (int64_t local = 0; local < count; ++local) {
            int64_t child = child_order[start + local];
            base_acc += static_cast<double>(grad_out[child * e + dim]);
        }
        grad_base[p * e + dim] = static_cast<scalar_t>(base_acc);
    }
    for (int64_t rr = 0; rr < r; ++rr) {
        double gate_acc = 0.0;
        double gate_val = static_cast<double>(gate[p * r + rr]);
        for (int64_t dim = 0; dim < e; ++dim) {
            double detail_acc = 0.0;
            double detail_val = static_cast<double>(detail[(p * r + rr) * e + dim]);
            for (int64_t local = 0; local < count; ++local) {
                int64_t child = child_order[start + local];
                double go = static_cast<double>(grad_out[child * e + dim]);
                double c = static_cast<double>(coeff[child * r + rr]);
                detail_acc += go * c * gate_val;
                gate_acc += go * c * detail_val;
                scale_acc += go * c * gate_val * detail_val;
            }
            grad_detail[(p * r + rr) * e + dim] = static_cast<scalar_t>(scale * detail_acc);
        }
        grad_gate[p * r + rr] = static_cast<scalar_t>(scale * gate_acc);
    }
    atomicAdd(grad_branch_scale, static_cast<scalar_t>(scale_acc));
}

}  // namespace

torch::Tensor grld_decode_forward_cuda(
    torch::Tensor base,
    torch::Tensor detail,
    torch::Tensor gate,
    torch::Tensor coeff,
    torch::Tensor child_to_parent,
    torch::Tensor branch_scale) {
    TORCH_CHECK(base.dim() == 2, "GRLD base must be [M,E]");
    TORCH_CHECK(detail.dim() == 3, "GRLD detail must be [M,R,E]");
    TORCH_CHECK(gate.dim() == 2, "GRLD gate must be [M,R]");
    TORCH_CHECK(coeff.dim() == 2, "GRLD coeff must be [N,R]");
    const int64_t m = base.size(0);
    const int64_t e = base.size(1);
    const int64_t r = detail.size(1);
    const int64_t n = coeff.size(0);
    TORCH_CHECK(detail.size(0) == m && detail.size(2) == e, "GRLD detail shape mismatch");
    TORCH_CHECK(gate.size(0) == m && gate.size(1) == r, "GRLD gate shape mismatch");
    TORCH_CHECK(coeff.size(1) == r, "GRLD coeff rank mismatch");
    TORCH_CHECK(child_to_parent.numel() == n, "GRLD child_to_parent length mismatch");
    auto out = torch::empty({n, e}, base.options());
    const int threads = 256;
    const int64_t total = n * e;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(base.scalar_type(), "grld_decode_forward_cuda", [&] {
        grld_decode_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            base.data_ptr<scalar_t>(),
            detail.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            coeff.data_ptr<scalar_t>(),
            child_to_parent.data_ptr<int64_t>(),
            branch_scale.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n,
            r,
            e);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

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
    torch::Tensor branch_scale) {
    const int64_t m = base.size(0);
    const int64_t e = base.size(1);
    const int64_t r = detail.size(1);
    const int64_t n = coeff.size(0);
    auto grad_base = torch::zeros_like(base);
    auto grad_detail = torch::zeros_like(detail);
    auto grad_gate = torch::zeros_like(gate);
    auto grad_coeff = torch::zeros_like(coeff);
    auto grad_branch_scale = torch::zeros_like(branch_scale);
    const int threads = 256;
    const int coeff_blocks = static_cast<int>(((n * r) + threads - 1) / threads);
    const int parent_blocks = static_cast<int>((m + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(base.scalar_type(), "grld_decode_backward_cuda", [&] {
        grld_grad_coeff_kernel<scalar_t><<<coeff_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_out.data_ptr<scalar_t>(),
            detail.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            child_to_parent.data_ptr<int64_t>(),
            branch_scale.data_ptr<scalar_t>(),
            grad_coeff.data_ptr<scalar_t>(),
            n,
            r,
            e);
        grld_parent_backward_kernel<scalar_t><<<parent_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_out.data_ptr<scalar_t>(),
            detail.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            coeff.data_ptr<scalar_t>(),
            child_order.data_ptr<int64_t>(),
            parent_start.data_ptr<int64_t>(),
            parent_count.data_ptr<int64_t>(),
            branch_scale.data_ptr<scalar_t>(),
            grad_base.data_ptr<scalar_t>(),
            grad_detail.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(),
            grad_branch_scale.data_ptr<scalar_t>(),
            m,
            r,
            e);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_base, grad_detail, grad_gate, grad_coeff, grad_branch_scale};
}
