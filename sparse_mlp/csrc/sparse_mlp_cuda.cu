#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "weight_cache.h"

// Forward declaration of the template
template <typename scalar_t>
__global__ void sparse_mlp_forward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ concat_weight,
    const scalar_t* __restrict__ active_down_weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size);

// Template specialization for half precision
template <>
__global__ void sparse_mlp_forward_cuda_kernel<at::Half>(
    const at::Half* __restrict__ input,
    const at::Half* __restrict__ concat_weight,
    const at::Half* __restrict__ active_down_weight,
    at::Half* __restrict__ output,
    at::Half* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;

    const __half* input_ptr = reinterpret_cast<const __half*>(input);
    const __half* weight_ptr = reinterpret_cast<const __half*>(concat_weight);
    const __half* down_ptr = reinterpret_cast<const __half*>(active_down_weight);
    __half* output_ptr = reinterpret_cast<__half*>(output);
    __half* combined_ptr = reinterpret_cast<__half*>(combined_buffer);

    // Get batch pointers
    const __half* batch_input = input_ptr + batch_idx * hidden_size;
    __half* batch_output = output_ptr + batch_idx * hidden_size;
    __half* batch_combined = combined_ptr + batch_idx * intermediate_size * 2;
    
    // Compute gate and up projections
    for (int i = tid; i < intermediate_size * 2; i += blockDim.x) {
        __half sum = __float2half(0.0f);
        for (int j = 0; j < hidden_size; j++) {
            sum = __hadd(sum, __hmul(batch_input[j], weight_ptr[i * hidden_size + j]));
        }
        batch_combined[i] = sum;
    }
    __syncthreads();
    
    // Apply activations and compute final output
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        __half sum = __float2half(0.0f);
        for (int j = 0; j < intermediate_size; j++) {
            const float gate_val = __half2float(batch_combined[j]);
            const __half gate = __float2half(1.0f / (1.0f + expf(-gate_val)));
            const __half up = batch_combined[j + intermediate_size];
            const __half down = down_ptr[i * intermediate_size + j];
            sum = __hadd(sum, __hmul(__hmul(gate, up), down));
        }
        batch_output[i] = sum;
    }
}

// Template specialization for float
template <>
__global__ void sparse_mlp_forward_cuda_kernel<float>(
    const float* __restrict__ input,
    const float* __restrict__ concat_weight,
    const float* __restrict__ active_down_weight,
    float* __restrict__ output,
    float* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;

    const float* batch_input = input + batch_idx * hidden_size;
    float* batch_output = output + batch_idx * hidden_size;
    float* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    for (int i = tid; i < intermediate_size * 2; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += batch_input[j] * concat_weight[i * hidden_size + j];
        }
        batch_combined[i] = sum;
    }
    __syncthreads();
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; j++) {
            const float gate = 1.0f / (1.0f + expf(-batch_combined[j]));
            const float up = batch_combined[j + intermediate_size];
            sum += gate * up * active_down_weight[i * intermediate_size + j];
        }
        batch_output[i] = sum;
    }
}

// Template specialization for double precision
template <>
__global__ void sparse_mlp_forward_cuda_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ concat_weight,
    const double* __restrict__ active_down_weight,
    double* __restrict__ output,
    double* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;

    const double* batch_input = input + batch_idx * hidden_size;
    double* batch_output = output + batch_idx * hidden_size;
    double* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    for (int i = tid; i < intermediate_size * 2; i += blockDim.x) {
        double sum = 0.0;
        for (int j = 0; j < hidden_size; j++) {
            sum += batch_input[j] * concat_weight[i * hidden_size + j];
        }
        batch_combined[i] = sum;
    }
    __syncthreads();
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        double sum = 0.0;
        for (int j = 0; j < intermediate_size; j++) {
            const double gate = 1.0 / (1.0 + exp(-batch_combined[j]));
            const double up = batch_combined[j + intermediate_size];
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
    torch::Tensor concat_weight = cache->get_concat_weight();
    torch::Tensor active_down_weight = cache->get_active_down_weight();
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    const auto intermediate_size = concat_weight.size(0) / 2;
    
    if (down_proj_buffer.size(0) != batch_size) {
        down_proj_buffer.resize_({batch_size, hidden_size});
    }
    if (combined_proj_buffer.size(0) != batch_size) {
        combined_proj_buffer.resize_({batch_size, concat_weight.size(0)});
    }

    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_forward_cuda", [&] {
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
    });
    return down_proj_buffer;
}