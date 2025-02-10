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
    
    void init(int64_t batch_size, const torch::Device& device = torch::kCPU, const c10::ScalarType dtype = torch::kFloat32) {
        clear();
        current_device = device;
        auto options = torch::TensorOptions()
            .device(current_device)
            .dtype(dtype);
        
        active_weights = torch::empty({2048, 3276}, options);
        active_downs = torch::empty({1638, 2048}, options);
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