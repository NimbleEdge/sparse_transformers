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

// Add device check utilities (only if not CPU-only build)
#ifndef CPU_ONLY
#include <c10/cuda/CUDAGuard.h>
#endif

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
    // Store original input shape for reshaping output later
    auto original_shape = input.sizes().vec();
    bool needs_reshape = input.dim() > 2;

    // Flatten input if it has more than 2 dimensions
    torch::Tensor input_2d;
    if (needs_reshape)
    {
        // Flatten all dimensions except the last one (hidden dimension)
        auto hidden_size = original_shape.back();
        auto total_batch_size = input.numel() / hidden_size;
        input_2d = input.view({total_batch_size, hidden_size});
    } else {
        input_2d = input;
    }

    const auto batch_size = input_2d.size(0);
    const auto hidden_size = input_2d.size(1);

    // Ensure output buffer is correctly sized
    // Check both dimensions to avoid resize warnings
    const int64_t expected_gate_dim = concat_weight.size(0);

    // For down_proj_buffer: [batch_size, hidden_size]
    if (down_proj_buffer.size(0) != batch_size || down_proj_buffer.size(1) != hidden_size)
        down_proj_buffer.resize_({batch_size, hidden_size});

    // For combined_proj_buffer: [batch_size, 2 * gate_size]
    if (combined_proj_buffer.size(0) != batch_size || combined_proj_buffer.size(1) != expected_gate_dim)
        combined_proj_buffer.resize_({batch_size, expected_gate_dim});

    // Optimal grain size for heavy matmul operations
    const int64_t num_threads = at::get_num_threads();
    int64_t grain_size = 1;
    
    if (batch_size > num_threads && num_threads > 0) {
        // Base calculation: create 2-4x more work chunks than threads
        const int64_t target_chunks = num_threads * 3;  // 3x oversubscription
        grain_size = std::max(int64_t(1), batch_size / target_chunks);
        
        // Cap at 32 for heavy operations (much smaller than 64 for light ops)
        const int64_t max_grain_matmul = 32;
        grain_size = std::min(grain_size, max_grain_matmul);
    }

    // Process each batch block in parallel
    at::parallel_for(0, batch_size, grain_size, [&](int64_t start, int64_t end)
                     {
        // Process blocks of batches instead of single items
        const int64_t block_size = end - start;
        const int64_t gate_size = concat_weight.size(0) / 2;
        
        // Get input block for this thread
        auto input_block = input_2d.slice(0, start, end);  // [block_size, hidden_size]
        
        // Get output buffer views for this block
        auto combined_proj_block = combined_proj_buffer.slice(0, start, end);  // [block_size, 2*gate_size]
        auto down_proj_block = down_proj_buffer.slice(0, start, end);  // [block_size, hidden_size]
        
        // Perform batch matrix multiplication for gate and up projections
        // This is more efficient than individual matmuls
        torch::matmul_out(combined_proj_block, input_block, concat_weight.t());
        
        // Split into gate and up projections
        auto gate_proj = combined_proj_block.narrow(1, 0, gate_size);  // [block_size, gate_size]
        auto up_proj = combined_proj_block.narrow(1, gate_size, gate_size);  // [block_size, gate_size]

        gate_proj.relu_();  // In-place relu
        gate_proj.mul_(up_proj);  // In-place element-wise multiplication
        
        // Final projection to output dimension
        torch::matmul_out(down_proj_block, gate_proj, active_down_weight.t()); });

    // Reshape output back to original shape if input was multi-dimensional
    return needs_reshape ? down_proj_buffer.view(original_shape) : down_proj_buffer;
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