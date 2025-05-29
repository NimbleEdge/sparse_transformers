# Fused Sparse C++ Kernels for Transformers

## Overview

The project implements the Sparse multiplication and fuses up/down projections in the MLP layers through low rank neural networks for weight activations. Work is based on the original contextual sparsity paper [ et al] and Apple’s LLM in a Flash [ et al].

- [x] Torch CPU kernels for fp16, fp32
- [x] Differential weight caching  and selection for dynamic sparsity
- [ ] CUDA kernels for Sparse Inferencing
- [ ] CPU kernels for int8, iint32, int64

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sparse LLM Inference Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│ Sparsity Selection                                              │
│   ├─ Hidden States → LoRA Projection (Importance Scoring)       │
│   ├─ Binary Mask Generation: (scores > threshold)               │
│   └─ Mask Normalization: Union across batch dimension           │
├─────────────────────────────────────────────────────────────────┤
│ Differential Weight Caching                                     │
│   ├─ Mask Change Detection: XOR with previous mask              │
│   ├─ Paired Replacement: Direct substitution algorithm          │
│   └─ Zero-Copy Tensor Views: torch::from_blob references        │
├─────────────────────────────────────────────────────────────────┤
│ Sparse Computation                                              │
│   ├─ Concatenated Gate+Up Projection (Fused Operation)          │
│   ├─ Element-wise Activation: σ(gate) ⊙ up                      │
│   └─ Sparse Down Projection: Only active intermediate dims      │
└─────────────────────────────────────────────────────────────────┘
```

**Keywords:** Large Language Models, Sparse Inference, Differential Weight Caching

## Current Performance Benchmarks
Since MLP layers account from 30% of total compute, a 5x gain in inference speed for MLP generates 1.6-1.8x overall speed gains.

### CPU Performance 
```
Sparse LLaMA 3.2 3B vs LLaMA 3.2 3B (on HuggingFace Implementation):

- Time to First Token (TTFT):  1.51× faster (1.209s → 0.803s)
- Output Generation Speed:     1.79× faster (0.7 → 1.2 tokens/sec)  
- Total Throughput:           1.78× faster (0.7 → 1.3 tokens/sec)
- Memory Usage:               26.4% reduction (13.25GB → 9.75GB)
```

### GPU Performance

```
Sparse LLaMA 3B vs Standard LLaMA 3B CUDA Results:

- Average time (Sparse): 0.021s
- Average time (Standard): 0.018s
- CUDA Speedups: 0.86x (WIP)
```

## Usage

### Quick Benchmark

```bash
# Run comprehensive benchmark

python run_benchmark.py \
    --device cpu \                              # Device: 'cpu' or 'cuda'
    --config configs/llama_skip_causal_3b.json \ # Model configuration
    --num_runs 50 \                            # Number of benchmark runs
    --verbose True                             # Detailed timing output

# Expected output:
# ⚡ TTFT Speedup: 1.51x
# 🚀 Output TPS Speedup: 1.79x  
# 📊 Total Throughput Speedup: 1.78x
```

## Implementation Details


### Paired Replacement with Differential Caching
_sparse_mlp/csrc/weight_cache.h_

The weight cache is a class that manages the active weights for the sparse MLP. It differentially updates the MLP tensor memory pool for the next token based on the predicted sparsity mask.

```cpp
class WeightCache {
    // Paired replacement algorithm for differential updates
    void update_active_weights(const torch::Tensor &mask)

};
```

**Performance Impact:**
- **6.7× faster cache updates**: 16.89ms → 8.36ms
- **Better cache locality**: Row major for Up Projection and Column major for Down Projection Matrices
- **Contiguous Memory Access**: Single memcpy for cache updates 

### Sparse MLP Inference
_sparse_mlp/csrc/sparse_mlp_op.cpp_

```python
sparse_mlp_forward(
    x.detach(), 
    self.weight_cache.get_concat_weight(),
    self.weight_cache.get_active_down_weight(),
    self.down_proj_buffer,
    self.combined_proj_buffer,
    "silu"
)
```

**Performance Impact:**
- **5× faster MLP inference**: 30.1ms → 6.02ms
- OpenMP parallelization with `torch::at::parallel_for`
- Bounded memory usage with weight cache memory pool

## Project Structure

```
├── sparse_mlp/                    # C++ extension module
│   ├── csrc/
│   │   ├── sparse_mlp_op.cpp     # Main CPU/CUDA dispatcher
│   │   ├── sparse_mlp_cuda.cu    # CUDA kernels
│   │   └── weight_cache.h        # Paired replacement caching
│   ├── __init__.py               # Python bindings
│   └── CMakeLists.txt            # Build configuration
├── src/models/llama/
│   ├── modelling_llama_skip.py   # Statistical sparsity model
│   └── configuration_llama_skip.py # Model configuration
├── tools/
│   └── component_timing.py       # Performance profiling
└── run_benchmark.py              # End-to-end benchmarks
```

## Installation


### Build C++ Extensions
```bash
# Clone repository
git clone https://github.com/nimbleedge/sparse_mlp.git
cd sparse_mlp

# Install in editable mode (builds C++ extensions automatically)
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import sparse_mlp; print('✅ Installation successful!')"
```

## Contributing

We welcome contributions! Areas of particular interest:
1. **Additional models**: Extend beyond LLaMA to other architectures
2. **Quantization**: Combine with INT8/FP16 optimizations
3. **Attention Kernels**: Implement Sparse Attention Kernels


