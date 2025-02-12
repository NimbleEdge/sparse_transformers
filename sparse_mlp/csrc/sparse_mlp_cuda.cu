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
    const int intermediate_idx = blockIdx.y * 32;  // Process warp-sized chunks
    const int batch_idx = blockIdx.z;
    
    if (hidden_idx >= hidden_size) return;

    // Get batch pointers with proper alignment
    const __half* batch_input = reinterpret_cast<const __half*>(input) + batch_idx * hidden_size;
    __half* batch_combined = reinterpret_cast<__half*>(combined_buffer) + batch_idx * intermediate_size * 2;
    const __half* weight_ptr = reinterpret_cast<const __half*>(concat_weight);
    
    
    // Process warp-sized chunk of intermediate dimension
    #pragma unroll 4
    for (int i = 0; i < 32 && intermediate_idx + i < intermediate_size; i++) {
        // Load and multiply in one step
        __half sum = __hmul(batch_input[hidden_idx], 
                           weight_ptr[(intermediate_idx + i) * hidden_size + hidden_idx]);
                      
        // Optimized warp reduction using butterfly pattern
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            __half other = __shfl_xor_sync(0xffffffff, sum, mask);
            sum = __hadd(sum, other);
        }
        // First thread in warp stores result
        if (tid == 0) {
            atomicAdd(&batch_combined[intermediate_idx + i], sum);
        }
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
    const int hidden_idx = blockIdx.y * 32;  // Process warp-sized chunks
    const int batch_idx = blockIdx.z;

    if (intermediate_idx >= intermediate_size) return;

    // Get batch pointers with proper alignment
    const __half* batch_combined = reinterpret_cast<const __half*>(combined_buffer) + batch_idx * intermediate_size * 2;
    const __half* down_ptr = reinterpret_cast<const __half*>(active_down_weight);
    __half* out_ptr = reinterpret_cast<__half*>(output)+ batch_idx * hidden_size + hidden_idx;
    
    const float gate_val = __half2float(batch_combined[intermediate_idx]);
    const __half up = batch_combined[intermediate_idx + intermediate_size];
    const __half gate = __float2half(1.0f / (1.0f + expf(-gate_val)));
    const __half mul_gate_up = __hmul(gate, up);
   
    // Compute using half precision
    #pragma unroll 4
    for (int i = 0; i < 32 && hidden_idx + i < hidden_size; i++) {
        const __half down = down_ptr[hidden_idx * intermediate_size + intermediate_idx + i];
        __half sum = __hmul(mul_gate_up, down);

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            __half other = __shfl_xor_sync(0xffffffff, sum, mask);
            sum = __hadd(sum, other);
        }
        if (tid == 0) {
            atomicAdd(&out_ptr[hidden_idx + i], sum);
        }
    }
}

torch::Tensor sparse_mlp_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& down_proj_buffer,
    torch::Tensor& combined_proj_buffer,
    const std::string& activation_fn) {
    
    // Create CUDA events for timing
    cudaEvent_t start, kernel1_start, kernel1_end, kernel2_end;
    cudaEventCreate(&start);
    cudaEventCreate(&kernel1_start);
    cudaEventCreate(&kernel1_end);
    cudaEventCreate(&kernel2_end);
    
    // // Allocate timing buffer
    // float* timing_buffer;
    // cudaMalloc(&timing_buffer, 4 * sizeof(float));
    
    auto cache = WeightCache::getInstance();
    
    // Record start time
    cudaEventRecord(start);
    
    torch::Tensor concat_weight = cache->get_concat_weight();
    torch::Tensor active_down_weight = cache->get_active_down_weight();
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    const auto intermediate_size = concat_weight.size(0) / 2;

    const int threads_per_block = 256;
    const int blocks_x = (hidden_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_x, 
              (intermediate_size + 31) / 32,  // Group by warps
              batch_size);
    dim3 block(threads_per_block, 1, 1);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    // Record kernel1 start
    cudaEventRecord(kernel1_start, stream);

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

    // // Record kernel1 end
    cudaEventRecord(kernel1_end, stream);
    const int blocks_intermediate = (intermediate_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid2(blocks_intermediate, 
              (hidden_size + 31) / 32,  // Group by warps
              batch_size);
    dim3 block2(threads_per_block, 1, 1);
    cudaStreamSynchronize(stream);

    // Launch second kernel
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

    // Record kernel2 end
    cudaEventRecord(kernel2_end, stream);
    cudaStreamSynchronize(stream);

    // // Calculate timing
    float setup_time, kernel1_time, kernel2_time;
    cudaEventElapsedTime(&setup_time, start, kernel1_start);
    cudaEventElapsedTime(&kernel1_time, kernel1_start, kernel1_end);
    cudaEventElapsedTime(&kernel2_time, kernel1_end, kernel2_end);

    printf("CUDA Kernel Timings:\n");
    printf("  Setup Time:      %.3f ms\n", setup_time);
    printf("  Combined Kernel: %.3f ms\n", kernel1_time);
    printf("  Output Kernel:   %.3f ms\n", kernel2_time);
    printf("  Total Time:      %.3f ms\n", setup_time + kernel1_time + kernel2_time);

    // // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(kernel1_start);
    cudaEventDestroy(kernel1_end);
    cudaEventDestroy(kernel2_end);
    // cudaFree(timing_buffer);

    return down_proj_buffer;
}