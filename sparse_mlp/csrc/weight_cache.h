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
    
    void init(int64_t batch_size, const torch::Device& device = torch::kCPU) {
        clear();
        current_device = device;
        auto options = torch::TensorOptions()
            .device(current_device)
            .dtype(torch::kFloat32);
        
        active_weights = torch::empty({2048, 3276}, options);
        active_downs = torch::empty({1638, 2048}, options);
        is_initialized = true;
    }
    
    void store(const torch::Tensor& concat_weights, const torch::Tensor& down) {
        // Move tensors to current device if needed
        active_weights = concat_weights.to(current_device);
        active_downs = down.to(current_device);
    }
    
    std::tuple<torch::Tensor, torch::Tensor> get() {
        return std::make_tuple(active_weights, active_downs);
    }
    
    torch::Device device() const {
        return current_device;
    }
}; 