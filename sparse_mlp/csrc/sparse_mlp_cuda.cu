#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "weight_cache.h"

// Helper function for half precision atomic add
__device__ __forceinline__ void atomicAdd_half(__half* address, __half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    
    do {
        assumed = old;
        __half_raw raw_val;
        raw_val.x = __half_as_short(val);
        unsigned int new_val;
        if ((size_t)address & 2) {
            new_val = (old & 0x0000FFFF) | (raw_val.x << 16);
        } else {
            new_val = (old & 0xFFFF0000) | raw_val.x;
        }
        old = atomicCAS(address_as_ui, assumed, new_val);
    } while (assumed != old);
}

// CUDA kernel for sparse MLP forward pass
template <typename scalar_t>
__global__ void sparse_mlp_forward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ concat_weight,
    const scalar_t* __restrict__ active_down_weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Get input and output pointers for this batch
    const scalar_t* batch_input = input + batch_idx * hidden_size;
    scalar_t* batch_output = output + batch_idx * hidden_size;
    scalar_t* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    // Compute gate and up projections
    for (int i = tid; i < intermediate_size * 2; i += blockDim.x) {
        scalar_t sum = 0;
        for (int j = 0; j < hidden_size; j++) {
            sum += batch_input[j] * concat_weight[i * hidden_size + j];
        }
        batch_combined[i] = sum;
    }
    __syncthreads();
    
    // Apply activations and compute final output
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        scalar_t sum = 0;
        for (int j = 0; j < intermediate_size; j++) {
            const scalar_t gate = 1.0f / (1.0f + exp(-batch_combined[j]));
            const scalar_t up = batch_combined[j + intermediate_size];
            sum += gate * up * active_down_weight[i * intermediate_size + j];
        }
        batch_output[i] = sum;
    }
}

torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn) {
    
    auto cache = WeightCache::getInstance();
    auto [concat_weight, active_down_weight] = cache->get();
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    const auto intermediate_size = concat_weight.size(0) / 2;
    
    // Ensure buffers are on CUDA and correctly sized
    down_proj_buffer = down_proj_buffer.to(input.device());
    combined_proj_buffer = combined_proj_buffer.to(input.device());
    
    if (down_proj_buffer.size(0) != batch_size) {
        down_proj_buffer.resize_({batch_size, hidden_size});
    }
    if (combined_proj_buffer.size(0) != batch_size) {
        combined_proj_buffer.resize_({batch_size, concat_weight.size(0)});
    }
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_forward_cuda", ([&] {
        sparse_mlp_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            concat_weight.data_ptr<scalar_t>(),
            active_down_weight.data_ptr<scalar_t>(),
            down_proj_buffer.data_ptr<scalar_t>(),
            combined_proj_buffer.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            intermediate_size
        );
    }));
    
    return down_proj_buffer;
}