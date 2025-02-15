#pragma once

#include <torch/extension.h>

class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    torch::Tensor active_weights; // Combined weights
    torch::Tensor active_downs;
    torch::Device current_device = torch::kCPU;
    
    // Delete copy/move operations
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
protected:
    WeightCache() = default;
    friend c10::intrusive_ptr<WeightCache>;

public:
    static c10::intrusive_ptr<WeightCache> getInstance() {
        static c10::intrusive_ptr<WeightCache> instance = 
            c10::make_intrusive<WeightCache>();
        return instance;
    }
    
    void clear() {
        active_weights = torch::Tensor();
        active_downs = torch::Tensor();
        is_initialized = false;
    }
    
    void init(const torch::Tensor& mask, int64_t hidden_size) {
        clear();
        current_device = mask.device();
        
        // Ensure sizes are aligned for half precision
        int64_t aligned_hidden_size = (hidden_size + 1) & ~1;
        int64_t aligned_sparse_size = (mask.size(1) + 1) & ~1;
        
        auto options = torch::TensorOptions()
            .device(current_device)
            .dtype(mask.scalar_type());
        
        active_weights = torch::empty({aligned_hidden_size, 2*aligned_sparse_size}, options);
        active_downs = torch::empty({aligned_sparse_size, aligned_hidden_size}, options);
        is_initialized = true;
    }
    
    void store(const torch::Tensor& concat_weights, const torch::Tensor& down) {
        active_weights = concat_weights.to(current_device);
        active_downs = down.to(current_device);
    }
    
    // Separate getters instead of structured bindings
    torch::Tensor get_concat_weight() const {
        return active_weights;
    }
    
    torch::Tensor get_active_down_weight() const {
        return active_downs;
    }
    
    torch::Device device() const {
        return current_device;
    }
}; 