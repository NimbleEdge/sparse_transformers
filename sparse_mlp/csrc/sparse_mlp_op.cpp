// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's parallel primitives
#include <ATen/ParallelOpenMP.h>

// For background tasks
#include <future>
#include <atomic>

// Add pybind11 and namespace
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Add weight cache class with layer and batch support TODO Make it Model instance dependent
class WeightCache : public torch::CustomClassHolder {
private:
    bool is_initialized = false;
    std::vector<torch::Tensor> active_gates;
    std::vector<torch::Tensor> active_ups;
    std::vector<torch::Tensor> active_downs;
    
    // Delete copy/move operations
    WeightCache(const WeightCache&) = delete;
    WeightCache& operator=(const WeightCache&) = delete;
    WeightCache(WeightCache&&) = delete;
    WeightCache& operator=(WeightCache&&) = delete;
    
protected:
    WeightCache() = default;
    friend c10::intrusive_ptr<WeightCache>;  // Allow make_intrusive to access constructor

public:
    static c10::intrusive_ptr<WeightCache> getInstance() {
        static c10::intrusive_ptr<WeightCache> instance = c10::make_intrusive<WeightCache>();
        return instance;
    }
    
    void clear() {
        active_gates = std::vector<torch::Tensor>();
        active_ups = std::vector<torch::Tensor>();
        active_downs = std::vector<torch::Tensor>();
        is_initialized = false;
    }
    
    void init(int64_t batch_size) {
        clear();
        active_gates = std::vector<torch::Tensor>(batch_size);
        active_ups = std::vector<torch::Tensor>(batch_size);
        active_downs = std::vector<torch::Tensor>(batch_size);
        is_initialized = true;
    }
    
    void store(int64_t batch_idx, 
               const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& down) {
        active_gates[batch_idx] = gate;
        active_ups[batch_idx] = up;
        active_downs[batch_idx] = down;
    }
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get(int64_t batch_idx) {
        auto gate = active_gates[batch_idx];
        auto up = active_ups[batch_idx];
        auto down = active_downs[batch_idx];
        
        if (!gate.defined() || !up.defined() || !down.defined()) {
            std::cout << "Error: Weights not initialized for batch " << batch_idx << std::endl;
            return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
        }
        
        return std::make_tuple(gate, up, down);
    }

    void store_gate(int64_t batch_idx, const torch::Tensor& gate) {
        active_gates[batch_idx] = gate;
    }
    
    void store_up(int64_t batch_idx, const torch::Tensor& up) {
        active_ups[batch_idx] = up;
    }
    
    void store_down(int64_t batch_idx, const torch::Tensor& down) {
        active_downs[batch_idx] = down;
    }
};

// Add background task manager
class BackgroundTaskManager {
private:
    std::future<void> current_task;
    std::atomic<bool> is_running{false};
    
    static BackgroundTaskManager* instance;
    BackgroundTaskManager() = default;
    
public:
    static BackgroundTaskManager* getInstance() {
        if (!instance) {
            instance = new BackgroundTaskManager();
        }
        return instance;
    }
    
    void start_task(std::function<void()> task) {
        if (is_running.load()) {
            if (current_task.valid()) {
                current_task.wait();
            }
        }
        
        is_running.store(true);
        current_task = std::async(std::launch::async, [this, task]() {
            task();
            is_running.store(false);
        });
    }
    
    void wait_if_running() {
        if (is_running.load() && current_task.valid()) {
            current_task.wait();
        }
    }
};

BackgroundTaskManager* BackgroundTaskManager::instance = nullptr;

void compute_active_weights(
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& mask) {
    
    auto task = [gate_weight, up_weight, down_weight, mask]() {
        int64_t batch_size = mask.size(0);
        WeightCache::getInstance()->init(batch_size);
        
        // Increase grain size and flatten the parallelism
        int64_t total_work = batch_size * 3;  // 3 operations per batch
        int64_t grain_size = std::max(int64_t(1), total_work / (8 * 4));
        at::parallel_for(0, total_work, grain_size, [&](int64_t start, int64_t end) {
            for (int64_t idx = start; idx < end; idx++) {
                int64_t batch_idx = idx / 3;
                int64_t op_idx = idx % 3;
                
                auto batch_mask = mask[batch_idx];
                auto active_indices = batch_mask.nonzero().squeeze();
                
                torch::Tensor result;
                if (op_idx == 0) {
                    result = gate_weight.index_select(0, active_indices).detach();
                    WeightCache::getInstance()->store_gate(batch_idx, result);
                } else if (op_idx == 1) {
                    result = up_weight.index_select(0, active_indices).detach();
                    WeightCache::getInstance()->store_up(batch_idx, result);
                } else {
                    result = down_weight.index_select(1, active_indices).detach();
                    WeightCache::getInstance()->store_down(batch_idx, result);
                }
            }
        });
    };
    
    BackgroundTaskManager::getInstance()->start_task(task);
}

// Modify sparse_mlp_forward to wait for weights if needed
torch::Tensor sparse_mlp_forward(torch::Tensor x, std::string act_fn_name) {
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(x.device())
        .layout(torch::kStrided);
    
    // Pre-allocate tensors for intermediate results
    std::vector<torch::Tensor> gate_activations(batch_size);
    std::vector<torch::Tensor> up_projections(batch_size);
    auto down_proj = torch::empty({batch_size, hidden_size}, options);
    
    BackgroundTaskManager::getInstance()->wait_if_running();
    
    // Phase 1: Compute gate activations and up projections in parallel
    int64_t total_work = batch_size * 2; // gate_proj and up_proj for each batch
    int64_t grain_size = std::max(int64_t(1), total_work / (8 * 4));
    at::parallel_for(0, total_work, grain_size, [&](int64_t start, int64_t end) {
        for (int64_t idx = start; idx < end; idx++) {
            int64_t batch_idx = idx / 2;
            bool is_gate = (idx % 2 == 0);
            
            auto [active_gate_weight, active_up_weight, active_down_weight] = 
                WeightCache::getInstance()->get(batch_idx);
            auto x_batch = x[batch_idx].view({1, hidden_size});
            
            if (is_gate) {
                // Compute gate activation
                auto gate_proj = torch::matmul(x_batch.detach(), active_gate_weight.t());
                gate_activations[batch_idx] = gate_proj * torch::sigmoid(gate_proj);
            } else {
                // Compute up projection
                up_projections[batch_idx] = torch::matmul(x_batch.detach(), active_up_weight.t());
            }
        }
    });
    
    // Phase 2: Combine results and compute final projection
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
            auto [_, __, active_down_weight] = WeightCache::getInstance()->get(batch_idx);
            
            // Combine gate activation and up projection
            auto gate_act = gate_activations[batch_idx] * up_projections[batch_idx];
            
            // Final projection
            down_proj[batch_idx] = torch::matmul(gate_act, active_down_weight.t())[0];
        }
    });
    
    return down_proj;
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
        .def("get", &WeightCache::get)
        .def("clear", &WeightCache::clear)
        .def("__repr__", [](const WeightCache&) {
            return "WeightCache(singleton)";
        });
}

// Register TorchScript operators
TORCH_LIBRARY(sparse_mlp, m) {
    m.def("forward", sparse_mlp_forward);
    m.def("compute_active_weights", compute_active_weights);
    m.class_<WeightCache>("WeightCache")
        .def_static("getInstance", &WeightCache::getInstance)
        .def("init", &WeightCache::init)
        .def("store", &WeightCache::store)
        .def("get", &WeightCache::get)
        .def("clear", &WeightCache::clear);
} 