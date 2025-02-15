#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "weight_cache.h"
#include "timer.h"

// Forward declarations with timing buffer
template <typename scalar_t>
__global__ void sparse_mlp_combined_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ concat_weight,
    scalar_t* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size);

template <typename scalar_t>
__global__ void sparse_mlp_output_cuda_kernel(
    const scalar_t* __restrict__ combined_buffer,
    const scalar_t* __restrict__ active_down_weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size);

// Template specializations
template <>
__global__ void sparse_mlp_combined_cuda_kernel<float>(
    const float* __restrict__ input,
    const float* __restrict__ concat_weight,
    float* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = threadIdx.z + blockIdx.z * blockDim.z;
    const int intermediate_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int hidden_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (batch_idx >= batch_size || intermediate_idx >= intermediate_size || hidden_idx >= hidden_size)
        return;

    // Get batch pointers
    const float* batch_input = input + batch_idx * hidden_size;
    float* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    // Compute sum for this thread
    float sum = batch_input[hidden_idx] * concat_weight[intermediate_idx * hidden_size + hidden_idx];
    
    // Warp reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Atomic add to global combined buffer
    if (threadIdx.x == 0) {
        atomicAdd(&batch_combined[intermediate_idx], sum);
    }
}

// Second kernel: compute output using combined values
template <>
__global__ void sparse_mlp_output_cuda_kernel<float>(
    const float* __restrict__ combined_buffer,
    const float* __restrict__ active_down_weight,
    float* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = threadIdx.z + blockIdx.z * blockDim.z;
    const int intermediate_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int hidden_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (batch_idx >= batch_size || intermediate_idx >= intermediate_size || hidden_idx >= hidden_size)
        return;

    // Get batch pointers
    const float* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    const float gate_val = batch_combined[intermediate_idx];
    const float gate = 1.0f / (1.0f + expf(-gate_val));
    const float up = batch_combined[intermediate_idx + intermediate_size];
    const float down = active_down_weight[hidden_idx * intermediate_size + intermediate_idx];
    const float val = gate * up * down;
    atomicAdd(&output[batch_idx * hidden_size + hidden_idx], val);
}

// First kernel for double
template <>
__global__ void sparse_mlp_combined_cuda_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ concat_weight,
    double* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = threadIdx.z + blockIdx.z * blockDim.z;
    const int intermediate_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int hidden_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (batch_idx >= batch_size || intermediate_idx >= intermediate_size || hidden_idx >= hidden_size)
        return;

    // Get batch pointers
    const double* batch_input = input + batch_idx * hidden_size;
    double* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    // Compute sum for this thread
    double sum = batch_input[hidden_idx] * concat_weight[intermediate_idx * hidden_size + hidden_idx];
    
    // Warp reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Atomic add to global combined buffer
    if (threadIdx.x == 0) {
        atomicAdd(&batch_combined[batch_idx * intermediate_size*2 + intermediate_idx], sum);
    }
}

// Second kernel for double
template <>
__global__ void sparse_mlp_output_cuda_kernel<double>(
    const double* __restrict__ combined_buffer,
    const double* __restrict__ active_down_weight,
    double* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int batch_idx = threadIdx.z + blockIdx.z * blockDim.z;
    const int intermediate_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int hidden_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (batch_idx >= batch_size || intermediate_idx >= intermediate_size || hidden_idx >= hidden_size)
        return;

    // Get batch pointers
    const double* batch_combined = combined_buffer + batch_idx * intermediate_size * 2;
    
    const double gate_val = batch_combined[intermediate_idx];
    const double gate = 1.0 / (1.0 + exp(-gate_val));
    const double up = batch_combined[intermediate_idx + intermediate_size];
    const double down = active_down_weight[hidden_idx * intermediate_size + intermediate_idx];
    const double val = gate * up * down;
    atomicAdd(&output[batch_idx * hidden_size + hidden_idx], val);
}

// First kernel for half precision
template <>
__global__ void sparse_mlp_combined_cuda_kernel<at::Half>(
    const at::Half* __restrict__ input,
    const at::Half* __restrict__ concat_weight,
    at::Half* __restrict__ combined_buffer,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int tid = threadIdx.x;
    const int hidden_idx = blockIdx.x * blockDim.x + tid;
    const int intermediate_idx = blockIdx.y * 16;
    const int batch_idx = blockIdx.z;
    const int lane_id = tid % 16;
    
    if (hidden_idx >= hidden_size || 2*intermediate_idx >= intermediate_size) return;

    __shared__ __half2 warp_sums[32];
    // Get batch pointers with proper alignment
    const __half2* batch_input = reinterpret_cast<const __half2*>(input) + batch_idx * hidden_size/2;
    __half2* batch_combined = reinterpret_cast<__half2*>(combined_buffer) + batch_idx * intermediate_size;
    const __half2* weight_ptr = reinterpret_cast<const __half2*>(concat_weight);
    __half2 input_pair = batch_input[hidden_idx];

    // Process warp-sized chunk of intermediate dimension
    #pragma unroll 8
    for (int i = 0; i < 16 && intermediate_idx + i*2 < intermediate_size; i+=2) {
        // Multiply both pairs at once
        __half2 sum = __hmul2(input_pair, weight_ptr[(intermediate_idx + i) * hidden_size/2 + hidden_idx]);
        __half2 sum2 = __hmul2(input_pair, weight_ptr[(intermediate_idx + i + 1) * hidden_size/2 + hidden_idx]);
        __half2 sum3 = __hmul2(input_pair, weight_ptr[(intermediate_idx + i + intermediate_size/2) * hidden_size/2 + hidden_idx]);
        __half2 sum4 = __hmul2(input_pair, weight_ptr[(intermediate_idx + i + intermediate_size/2 + 1) * hidden_size/2 + hidden_idx]);
        
        // Optimized warp reduction using butterfly pattern with half2
        #pragma unroll
        for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
            sum = __hadd2(sum, __shfl_xor_sync(0xffffffff, sum, mask));
            sum2 = __hadd2(sum2, __shfl_xor_sync(0xffffffff, sum2, mask));
            sum3 = __hadd2(sum3, __shfl_xor_sync(0xffffffff, sum3, mask));
            sum4 = __hadd2(sum4, __shfl_xor_sync(0xffffffff, sum4, mask));
        }
        
        // Store results to shared memory
        if (tid == 0) {
            warp_sums[i] = sum;               
            warp_sums[i+1] = sum2;
            warp_sums[i+16] = sum3;
            warp_sums[i+17] = sum4;
        }
    }
    
    __syncwarp();

    // Have first warp do the atomic adds
    if (tid < 16 && intermediate_idx + lane_id < intermediate_size) {
        atomicAdd(&batch_combined[intermediate_idx + lane_id], warp_sums[lane_id]);
        atomicAdd(&batch_combined[intermediate_idx + intermediate_size/2 + lane_id], warp_sums[lane_id+32]);
    }
}

// Second kernel for half precision
template <>
__global__ void sparse_mlp_output_cuda_kernel<at::Half>(
    const at::Half* __restrict__ combined_buffer,
    const at::Half* __restrict__ active_down_weight,
    at::Half* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size) {
    
    const int tid = threadIdx.x;
    const int intermediate_idx = blockIdx.x * blockDim.x + tid;
    const int hidden_idx = blockIdx.y * 16;
    const int batch_idx = blockIdx.z;
    const int lane_id = tid % 16;

    if (2*intermediate_idx >= intermediate_size) return;

    // Shared memory for partial sums and intermediate values
    __shared__ __half2 shared_sums[16];
    __half2 gate_up_cache;  // Cache gate/up values for reuse
    
    // Get batch pointers with proper alignment
    const __half2* batch_combined2 = reinterpret_cast<const __half2*>(combined_buffer) + 
                                    batch_idx * intermediate_size;
    const __half2* down_ptr2 = reinterpret_cast<const __half2*>(active_down_weight);
    __half2* out_ptr2 = reinterpret_cast<__half2*>(output) + (batch_idx * (hidden_size) + hidden_idx)/2;

    // Load and process gate/up values - cache in shared memory
    __half2 combined = batch_combined2[intermediate_idx];
    float2 gate_val = __half22float2(combined);
    __half2 gate = __float2half2_rn(1.0f / (1.0f + expf(-gate_val.x)));
    __half2 up = batch_combined2[intermediate_idx+intermediate_size/2];
    gate_up_cache = __hmul2(gate, up);

    // Process 4 elements per iteration using 2x half2
    #pragma unroll 8
    for (int i = 0; i < 16 && hidden_idx + i*2 < hidden_size; i += 2) {
        // Load two pairs of down weights
        __half2 down1 = down_ptr2[(hidden_idx + i) * (intermediate_size/2) + intermediate_idx];
        __half2 down2 = down_ptr2[(hidden_idx + i + 1) * (intermediate_size/2) + intermediate_idx];
        
        // Multiply with cached gate_up values
        __half2 sum1 = __hmul2(gate_up_cache, down1);
        __half2 sum2 = __hmul2(gate_up_cache, down2);
        
        // Warp reduction for both pairs
        #pragma unroll
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            sum1 = __hadd2(sum1, __shfl_xor_sync(0xffffffff, sum1, offset));
            sum2 = __hadd2(sum2, __shfl_xor_sync(0xffffffff, sum2, offset));
        }
        
        // Store results
        if (tid == 0) {
            shared_sums[i] = sum1;
            shared_sums[i+1] = sum2;
        }
    }
    
    __syncwarp();
    
    // Coalesced writes to global memory - 4 elements at a time
    if (tid < 8 && hidden_idx + lane_id*2 < hidden_size) {
        atomicAdd(&out_ptr2[lane_id*2], shared_sums[lane_id*2]);
        atomicAdd(&out_ptr2[lane_id*2+1], shared_sums[lane_id*2+1]);
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

    const int threads_per_block = 256;
    const int blocks_x = (hidden_size + threads_per_block - 1) / (2*threads_per_block);
    
    dim3 grid(blocks_x, 
            (intermediate_size + 15) / 16,  // Group by warps
              batch_size);
    dim3 block(threads_per_block, 1, 1);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    // Launch first kernel with timing buffer
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_combined_cuda", [&] {
        sparse_mlp_combined_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            concat_weight.data_ptr<scalar_t>(),
            combined_proj_buffer.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            intermediate_size
        );
    });

    const int blocks_intermediate = (intermediate_size + threads_per_block - 1) / (2*threads_per_block);
    
    dim3 grid2(blocks_intermediate, 
              (hidden_size + 15) / 16,  // Group by warps
              batch_size);
    dim3 block2(threads_per_block, 1, 1);
    cudaStreamSynchronize(stream);
    
    // Launch second kernel with timing buffer
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_mlp_output_cuda", [&] {
        sparse_mlp_output_cuda_kernel<scalar_t><<<grid2, block2, 0, stream>>>(
            combined_proj_buffer.data_ptr<scalar_t>(),
            active_down_weight.data_ptr<scalar_t>(),
            down_proj_buffer.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            intermediate_size
        );
    });

    return down_proj_buffer;
}