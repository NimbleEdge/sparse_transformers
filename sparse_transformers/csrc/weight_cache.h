#pragma once

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <immintrin.h> // For SIMD operations

class WeightCache : public torch::CustomClassHolder
{
private:
    // Define deleter as a struct to avoid std::function overhead
    struct AlignedDeleter
    {
        void operator()(float *ptr) const
        {
            free(ptr);
        }
    };

    bool is_initialized = false;

    // Memory pools for all weight data (cache-aligned)
    std::unique_ptr<float[], AlignedDeleter> gate_memory_pool;
    std::unique_ptr<float[], AlignedDeleter> up_memory_pool;
    std::unique_ptr<float[], AlignedDeleter> down_memory_pool_transposed; // Store transposed for fast row access

    // Matrix dimensions
    int64_t hidden_dim = 0;
    int64_t sparse_dim = 0;
    int64_t gate_row_size = 0;
    int64_t up_row_size = 0;
    int64_t down_row_size = 0; // This becomes hidden_dim after transpose

    torch::ScalarType dtype;
    torch::Device current_device = torch::kCPU;

    // Currently active indices (maintained in order for contiguous access)
    std::vector<int64_t> active_indices;

    // Mapping from active_index to position in active_indices for O(1) lookup
    std::unordered_map<int64_t, size_t> index_to_position;

    // Contiguous buffers for active data (always packed)
    std::unique_ptr<float[], AlignedDeleter> active_gate_buffer;
    std::unique_ptr<float[], AlignedDeleter> active_up_buffer;
    std::unique_ptr<float[], AlignedDeleter> active_down_buffer;

    // Current mask for differential updates
    torch::Tensor current_mask;

    // Cached active weight tensors - use from_blob to reference our buffers directly
    torch::Tensor active_weights_cache;
    torch::Tensor active_downs_cache;
    bool cache_valid = false;

    // Max expected active indices (dynamic based on intermediate_size)
    size_t max_active_indices = 0;

    // Cache-aligned memory allocation
    static void *aligned_alloc_wrapper(size_t size)
    {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0)
        { // 64-byte alignment for cache lines
            throw std::bad_alloc();
        }
        return ptr;
    }

    // Find differential changes between masks using PyTorch operations
    struct MaskDiff
    {
        std::vector<int64_t> added_indices;
        std::vector<int64_t> removed_indices;
    };

    MaskDiff compute_mask_diff(const torch::Tensor &old_mask, const torch::Tensor &new_mask)
    {
        MaskDiff diff;

        // Use PyTorch operations for efficient mask comparison
        auto added_mask = new_mask & (~old_mask);   // new & ~old = added
        auto removed_mask = old_mask & (~new_mask); // old & ~new = removed

        // Get indices of added and removed elements
        auto added_indices_tensor = torch::nonzero(added_mask).squeeze(-1);
        auto removed_indices_tensor = torch::nonzero(removed_mask).squeeze(-1);

        // Convert to std::vector
        if (added_indices_tensor.numel() > 0)
        {
            auto added_data = added_indices_tensor.data_ptr<int64_t>();
            diff.added_indices.assign(added_data, added_data + added_indices_tensor.numel());
        }

        if (removed_indices_tensor.numel() > 0)
        {
            auto removed_data = removed_indices_tensor.data_ptr<int64_t>();
            diff.removed_indices.assign(removed_data, removed_data + removed_indices_tensor.numel());
        }

        return diff;
    }

    // Rebuild tensors using from_blob to reference our contiguous buffers
    void rebuild_tensor_views()
    {
        const size_t num_active = active_indices.size();

        if (num_active == 0)
        {
            auto options = torch::TensorOptions().device(current_device).dtype(dtype);
            active_weights_cache = torch::empty({0, hidden_dim}, options);
            active_downs_cache = torch::empty({0, hidden_dim}, options);
            return;
        }

        // Create gate tensor directly from buffer
        auto gate_tensor = torch::from_blob(active_gate_buffer.get(),
                                            {static_cast<int64_t>(num_active), gate_row_size},
                                            torch::TensorOptions().dtype(dtype));

        // Create up tensor directly from buffer
        auto up_tensor = torch::from_blob(active_up_buffer.get(),
                                          {static_cast<int64_t>(num_active), up_row_size},
                                          torch::TensorOptions().dtype(dtype));

        // Create down tensor directly from buffer and transpose
        auto down_tensor_packed = torch::from_blob(active_down_buffer.get(),
                                                   {static_cast<int64_t>(num_active), hidden_dim},
                                                   torch::TensorOptions().dtype(dtype));
        auto down_tensor = down_tensor_packed.t(); // [hidden_dim, num_active]

        // Concatenate and move to target device
        active_weights_cache = torch::cat({gate_tensor, up_tensor}, 0).to(current_device);
        active_downs_cache = down_tensor.to(current_device);
    }

public:
    WeightCache(const torch::Tensor &init_mask, int64_t hidden_size,
                const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
                const torch::Tensor &down_weight)
    {
        init(init_mask, hidden_size, gate_weight, up_weight, down_weight);
    }

    void init(const torch::Tensor &init_mask, int64_t hidden_size,
              const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
              const torch::Tensor &down_weight)
    {

        current_device = gate_weight.device();
        dtype = gate_weight.scalar_type();

        // Store dimensions
        hidden_dim = hidden_size;
        sparse_dim = gate_weight.size(0);
        max_active_indices = init_mask.sum().item<int64_t>();
        gate_row_size = gate_weight.size(1);
        up_row_size = up_weight.size(1);
        down_row_size = hidden_dim; // After transpose: [intermediate_size, hidden_size]

        // Allocate cache-aligned memory pools
        const size_t gate_total_size = sparse_dim * gate_row_size;
        const size_t up_total_size = sparse_dim * up_row_size;
        const size_t down_total_size = sparse_dim * hidden_dim; // Transposed shape

        gate_memory_pool = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(gate_total_size * sizeof(float))));
        up_memory_pool = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(up_total_size * sizeof(float))));
        down_memory_pool_transposed = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(down_total_size * sizeof(float))));

        // Pre-allocate contiguous buffers for active weights
        active_gate_buffer = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(max_active_indices * gate_row_size * sizeof(float))));
        active_up_buffer = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(max_active_indices * up_row_size * sizeof(float))));
        active_down_buffer = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(max_active_indices * hidden_dim * sizeof(float))));

        // Initialize differential update tracking
        index_to_position.reserve(max_active_indices);

        // Copy weights to memory pools
        auto gate_cpu = gate_weight.to(torch::kCPU).contiguous();
        auto up_cpu = up_weight.to(torch::kCPU).contiguous();
        auto down_cpu = down_weight.to(torch::kCPU).contiguous();

        // Copy gate and up weights directly (row-major format)
        std::memcpy(gate_memory_pool.get(), gate_cpu.data_ptr<float>(), gate_total_size * sizeof(float));
        std::memcpy(up_memory_pool.get(), up_cpu.data_ptr<float>(), up_total_size * sizeof(float));

        // Transpose down matrix during copy: [hidden_size, intermediate_size] -> [intermediate_size, hidden_size]
        auto down_data = down_cpu.data_ptr<float>();
        for (int64_t i = 0; i < sparse_dim; ++i)
        {
            for (int64_t j = 0; j < hidden_dim; ++j)
            {
                down_memory_pool_transposed[i * hidden_dim + j] = down_data[j * sparse_dim + i];
            }
        }

        is_initialized = true;
        current_mask = torch::zeros(sparse_dim, torch::TensorOptions().dtype(torch::kBool).device(current_device));

        // Initialize with mask
        update_active_weights(init_mask);
    }

    void update_active_weights(const torch::Tensor &mask)
    {
        if (!is_initialized)
            return;

        // Compute diff with normalization handled internally
        auto diff = compute_mask_diff(current_mask, mask);

        // Early exit if no changes - avoid all processing work!
        if (diff.added_indices.empty() && diff.removed_indices.empty())
        {
            return;
        }

        // Optimized single-pass removal+addition logic
        const size_t num_removals = diff.removed_indices.size();
        const size_t num_additions = diff.added_indices.size();
        const size_t pairs_to_process = std::min(num_removals, num_additions);

        // First pass: Pair removals with additions for direct replacement (most cache-efficient)
        for (size_t i = 0; i < pairs_to_process; ++i)
        {
            int64_t removed_idx = diff.removed_indices[i];
            int64_t added_idx = diff.added_indices[i];

            auto it = index_to_position.find(removed_idx);
            if (it != index_to_position.end())
            {
                size_t pos = it->second;

                // Direct replacement - copy new data over old position (single memcpy per matrix!)
                std::memcpy(active_gate_buffer.get() + pos * gate_row_size,
                            gate_memory_pool.get() + added_idx * gate_row_size,
                            gate_row_size * sizeof(float));

                std::memcpy(active_up_buffer.get() + pos * up_row_size,
                            up_memory_pool.get() + added_idx * up_row_size,
                            up_row_size * sizeof(float));

                std::memcpy(active_down_buffer.get() + pos * hidden_dim,
                            down_memory_pool_transposed.get() + added_idx * hidden_dim,
                            hidden_dim * sizeof(float));

                // Update tracking - remove old, add new at same position
                index_to_position.erase(it);
                active_indices[pos] = added_idx;
                index_to_position[added_idx] = pos;
            }
        }

        // Handle remaining additions (if more additions than removals)
        for (size_t i = pairs_to_process; i < num_additions; ++i)
        {
            int64_t added_idx = diff.added_indices[i];
            size_t new_pos = active_indices.size();

            if (new_pos >= max_active_indices)
            {
                continue; // Skip if buffer full
            }

            // Append to end
            std::memcpy(active_gate_buffer.get() + new_pos * gate_row_size,
                        gate_memory_pool.get() + added_idx * gate_row_size,
                        gate_row_size * sizeof(float));

            std::memcpy(active_up_buffer.get() + new_pos * up_row_size,
                        up_memory_pool.get() + added_idx * up_row_size,
                        up_row_size * sizeof(float));

            std::memcpy(active_down_buffer.get() + new_pos * hidden_dim,
                        down_memory_pool_transposed.get() + added_idx * hidden_dim,
                        hidden_dim * sizeof(float));

            // Update tracking
            active_indices.push_back(added_idx);
            index_to_position[added_idx] = new_pos;
        }

        // Handle remaining removals (if more removals than additions)
        for (size_t i = pairs_to_process; i < num_removals; ++i)
        {
            int64_t removed_idx = diff.removed_indices[i];
            auto it = index_to_position.find(removed_idx);
            if (it != index_to_position.end())
            {
                size_t pos_to_remove = it->second;
                size_t last_pos = active_indices.size() - 1;

                if (pos_to_remove != last_pos)
                {
                    // Move last element to fill gap
                    int64_t last_idx = active_indices[last_pos];

                    std::memcpy(active_gate_buffer.get() + pos_to_remove * gate_row_size,
                                active_gate_buffer.get() + last_pos * gate_row_size,
                                gate_row_size * sizeof(float));

                    std::memcpy(active_up_buffer.get() + pos_to_remove * up_row_size,
                                active_up_buffer.get() + last_pos * up_row_size,
                                up_row_size * sizeof(float));

                    std::memcpy(active_down_buffer.get() + pos_to_remove * hidden_dim,
                                active_down_buffer.get() + last_pos * hidden_dim,
                                hidden_dim * sizeof(float));

                    // Update tracking
                    active_indices[pos_to_remove] = last_idx;
                    index_to_position[last_idx] = pos_to_remove;
                }

                // Remove last element
                active_indices.pop_back();
                index_to_position.erase(it);
            }
        }

        // Rebuild tensor views using from_blob (no copying!)
        rebuild_tensor_views();
        cache_valid = true;
        current_mask = mask.clone();
    }

    // Getters remain the same
    torch::Tensor get_concat_weight() const
    {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_weights_cache;
    }

    torch::Tensor get_active_down_weight() const
    {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_downs_cache;
    }

    size_t get_num_active() const
    {
        return active_indices.size();
    }

    // Destructor - no manual cleanup needed with smart pointers
    ~WeightCache() = default;
};