// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's OpenMP wrapper
#include <ATen/ParallelOpenMP.h>

// Add pybind11 and namespace
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Add required headers
#include <future>
#include <thread>
#include <mutex>

// Add device check utilities
#include <c10/cuda/CUDAGuard.h>

// Add custom headers
#include "weight_cache.h"
#include "approx_topk.h"

// Forward declarations of CPU/CUDA implementations
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor &input,
    const torch::Tensor &concat_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &combined_proj_buffer,
    const std::string &activation_fn);

#ifdef WITH_CUDA
torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor &input,
    const torch::Tensor &concat_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &combined_proj_buffer,
    const std::string &activation_fn);
#endif

// Main dispatch function
torch::Tensor sparse_mlp_forward(
    const torch::Tensor &input,
    const torch::Tensor &concat_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &combined_proj_buffer,
    const std::string &activation_fn)
{

    // Check if input is on CUDA and dispatch accordingly
    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        return sparse_mlp_forward_cuda(input, concat_weight, active_down_weight, down_proj_buffer, combined_proj_buffer, activation_fn);
#else
        AT_ERROR("CUDA not available - cannot run on GPU");
#endif
    }
    else
    {
        return sparse_mlp_forward_cpu(input, concat_weight, active_down_weight, down_proj_buffer, combined_proj_buffer, activation_fn);
    }
}

// CPU implementation
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor &input,
    const torch::Tensor &concat_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &combined_proj_buffer,
    const std::string &activation_fn)
{

    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);

    // Ensure output buffer is correctly sized
    if (down_proj_buffer.size(0) != batch_size)
    {
        down_proj_buffer.resize_({batch_size, hidden_size});
    }
    if (combined_proj_buffer.size(0) != batch_size)
    {
        combined_proj_buffer.resize_({batch_size, 2 * int(concat_weight.size(0))});
    }

    // Process each batch item in parallel
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end)
                     {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
            int64_t gate_size = concat_weight.size(0) / 2;
            auto x_batch = input[batch_idx].unsqueeze(0).detach();
            
            // Single matmul for both gate and up projections
            auto proj_view = combined_proj_buffer[batch_idx].unsqueeze(0).narrow(1, 0, concat_weight.size(0));
            torch::matmul_out(proj_view, x_batch, concat_weight.t());
            
            // Split result into gate and up projections
            auto gate_proj = proj_view.narrow(1, 0, gate_size);
            auto up_proj = proj_view.narrow(1, gate_size, gate_size);
            
            // Apply activations
            gate_proj.mul_(torch::sigmoid(gate_proj));
            gate_proj.mul_(up_proj);
            
            // Final projection
            down_proj_buffer[batch_idx] = torch::matmul(gate_proj, active_down_weight.t())[0];
        } });
    return down_proj_buffer;
}

// Register TorchScript custom classes and operators
TORCH_LIBRARY(sparse_mlp, m)
{
    // Register the optimized weight cache
    m.class_<WeightCache>("WeightCache")
        .def(torch::init<const torch::Tensor &, int64_t, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &>())
        .def("update_active_weights", &WeightCache::update_active_weights)
        .def("get_concat_weight", &WeightCache::get_concat_weight)
        .def("get_active_down_weight", &WeightCache::get_active_down_weight);

    // Register sparse MLP operator
    m.def("forward", sparse_mlp_forward);

    // Register Count-Min Sketch approximate top-k threshold operator
    m.def("approx_topk_threshold", approx_topk_threshold);
}