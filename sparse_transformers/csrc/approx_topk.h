#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/Parallel.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

// Count-Min Sketch inspired method - O(n) time complexity with parallel batch processing
torch::Tensor approx_topk_threshold(
    const torch::Tensor &scores,
    int64_t k)
{
    TORCH_CHECK(scores.dim() == 2, "Input scores must be 2D tensor [batch_size, features]");
    TORCH_CHECK(k > 0, "k must be positive");

    auto batch_size = scores.size(0);
    auto feature_size = scores.size(1);

    TORCH_CHECK(k <= feature_size, "k cannot be larger than feature size");

    auto options = torch::TensorOptions().dtype(scores.dtype()).device(scores.device());
    auto threshold = torch::zeros({batch_size, 1}, options);

    // Sketch parameters
    const int num_sketches = 4;
    const int sketch_width = std::min(1024L, feature_size / 4);

    // Standard C++ hash function
    std::hash<int64_t> hasher;

    // Process each batch item in parallel using at::parallel_for
    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "approx_topk_count_min_sketch", [&]
                               {
        auto scores_accessor = scores.accessor<scalar_t, 2>();
        auto threshold_accessor = threshold.accessor<scalar_t, 2>();
        
        // Parallel processing over batch dimension
        // Use grain_size of 1 for fine-grained parallelism
        at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
            for (int64_t batch_idx = start; batch_idx < end; ++batch_idx) {
                // Initialize sketches with negative infinity (thread-local)
                std::vector<std::vector<scalar_t>> sketches(num_sketches, 
                    std::vector<scalar_t>(sketch_width, -std::numeric_limits<scalar_t>::infinity()));
                
                // Update sketches with maximum values at hash positions
                for (int sketch_idx = 0; sketch_idx < num_sketches; ++sketch_idx) {
                    for (int64_t feature_idx = 0; feature_idx < feature_size; ++feature_idx) {
                        // Use different hash functions for each sketch by combining with sketch_idx
                        int64_t combined_key = sketch_idx * feature_size + feature_idx;
                        int64_t hash_pos = hasher(combined_key) % sketch_width;
                        
                        scalar_t value = scores_accessor[batch_idx][feature_idx];
                        sketches[sketch_idx][hash_pos] = std::max(sketches[sketch_idx][hash_pos], value);
                    }
                }
                
                // Collect all sketch values (thread-local)
                std::vector<scalar_t> all_sketch_values;
                for (const auto& sketch : sketches) {
                    for (scalar_t val : sketch) {
                        if (val != -std::numeric_limits<scalar_t>::infinity()) {
                            all_sketch_values.push_back(val);
                        }
                    }
                }
                
                if (!all_sketch_values.empty()) {
                    // Find approximate threshold
                    int64_t sketch_k = std::max(1L, static_cast<int64_t>(k * all_sketch_values.size() / feature_size));
                    sketch_k = std::min(sketch_k, static_cast<int64_t>(all_sketch_values.size()));
                    
                    std::nth_element(all_sketch_values.begin(), 
                                   all_sketch_values.begin() + sketch_k - 1, 
                                   all_sketch_values.end(), 
                                   std::greater<scalar_t>());
                    
                    // Apply adjustment factor for approximation error
                    scalar_t adjustment_factor = 0.9;
                    threshold_accessor[batch_idx][0] = all_sketch_values[sketch_k - 1] * adjustment_factor;
                } else {
                    threshold_accessor[batch_idx][0] = 0.0;
                }
            }
        }); });

    return threshold;
}