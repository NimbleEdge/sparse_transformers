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


void compute_active_weights(
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask) {
    int64_t batch_size = mask.size(0);
    int64_t sparse_size = mask.size(1);
    WeightCache::getInstance()->init(mask, gate_weight.size(1));
    auto active_gate = gate_weight.narrow(0, 0, sparse_size).detach();
    auto active_up = up_weight.narrow(0, 0, sparse_size).detach();
    // Concatenate gate and up weights
    auto concat_weights = torch::cat({active_gate, active_up}, 0);
    auto active_down = down_weight.narrow(1, 0, sparse_size).detach();
    
    WeightCache::getInstance()->store(concat_weights, active_down);
}

// Forward declarations of CPU/CUDA implementations
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn);

#ifdef WITH_CUDA
torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn);
#endif

// Main dispatch function
torch::Tensor sparse_mlp_forward(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn) {
    
    // Check if input is on CUDA and dispatch accordingly
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
            return sparse_mlp_forward_cuda(input, down_proj_buffer, combined_proj_buffer, activation_fn);
        #else
            AT_ERROR("CUDA not available - cannot run on GPU");
        #endif
    } else {
        return sparse_mlp_forward_cpu(input, down_proj_buffer, combined_proj_buffer, activation_fn);
    }
}

// CPU implementation
torch::Tensor sparse_mlp_forward_cpu(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn) {
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    
    // Ensure output buffer is correctly sized
    if (down_proj_buffer.size(0) != batch_size) {
        down_proj_buffer.resize_({batch_size, hidden_size});
    }
    
    // Process each batch item in parallel
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
            auto cache = WeightCache::getInstance();
            torch::Tensor concat_weight = cache->get_concat_weight();
            torch::Tensor active_down_weight = cache->get_active_down_weight();
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
        }
    });
    return down_proj_buffer;
}

// Register operators and expose WeightCache to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_mlp_forward, "Sparse MLP forward");
    m.def("compute_active_weights", &compute_active_weights, "Compute active weights");
    
    // Expose WeightCache class
    py::class_<WeightCache, c10::intrusive_ptr<WeightCache>>(m, "WeightCache")
        .def_static("getInstance", &WeightCache::getInstance)
        .def("init", &WeightCache::init)
        .def("store", &WeightCache::store)
        .def("get_concat_weight", &WeightCache::get_concat_weight)
        .def("get_active_down_weight", &WeightCache::get_active_down_weight)
        .def("clear", &WeightCache::clear)
        .def("__repr__", [](const WeightCache&) {
            return "WeightCache(singleton)";
        });
}

// Register TorchScript operators
TORCH_LIBRARY(sparse_mlp, m) {
    m.def("forward", sparse_mlp_forward);
    m.def("compute_active_weights", compute_active_weights);
} 