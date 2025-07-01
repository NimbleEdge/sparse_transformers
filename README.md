[![Join us on Discord](https://img.shields.io/badge/Join%20us-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/y8WkMncstk)

# Fused Sparse C++ Kernels for Transformers

## Overview

The project implements sparse multiplication and fuses up/down projections in the MLP layers through low rank weight activations. 
Work is based on [Deja Vu](https://arxiv.org/abs/2310.17157) and Apple's [LLM in a Flash](https://arxiv.org/abs/2312.11514).

### Benefits
- **1.6-1.8x overall gain in TTFT and TPS** (4-5x gain in MLP Inference)
- **26.4%** reduction in memory usage
- **6.7×** faster index selection and replacement for weight caching


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

## Performance Benchmarks
State of Implementation:
- [x] Torch CPU kernels for fp16, fp32
- [x] Differential weight caching  and selection for dynamic sparsity
- [ ] CUDA kernels for Sparse Inferencing
- [ ] CPU kernels for int8, int32, int64

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
Sparse LLaMA 3.2 3B vs Standard LLaMA 3.2 3B CUDA Results (on HuggingFace Implementation):

- Average time (Sparse): 0.021s
- Average time (Standard): 0.018s
- CUDA Speedups: 0.86x (WIP)
```

## Usage

### Quick Benchmark

```bash
# Run comprehensive benchmark

python benchmark.py \
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
_sparse_transformers/csrc/weight_cache.h_

The weight cache is a class that manages the active weights for the sparse MLP. It differentially updates the MLP tensor memory pool for the next token based on the predicted sparsity mask.

```cpp
class WeightCache {
    // Paired replacement algorithm for differential updates
    void update_active_weights(const torch::Tensor &mask)

};
```

**Performance Impact:**
- **6.7× faster cache updates**: 29.89ms (naive `index_select`) → 4.46ms (paired replacement)
- **Better cache locality**: Row major for Up Projection and Column major for Down Projection Matrices
- **Contiguous Memory Access**: Single memcpy for cache updates 

### Sparse MLP Inference
_sparse_transformers/csrc/sparse_mlp_op.cpp_

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
- **5× faster CPU MLP inference**: 30.1ms → 6.02ms
- OpenMP parallelization with `torch::at::parallel_for`
- Bounded memory usage with weight cache memory pool

## Project Structure

```
├── sparse_transformers/                    # C++ extension module
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
git clone https://github.com/nimbleedge/sparse_transformers.git
cd sparse_transformers
```

Set up conda environment and install dependencies
```bash
conda create -n sparse_transformers python=3.10
conda activate sparse_transformers
```

Install torch dependencies from [requirements.txt](requirements.txt#L2)

```bash
# Install in editable mode (builds C++ extensions automatically)
pip install -r requirements.txt
pip install -e .                 # Auto-detect (prefer GPU if available)
pip install -e . --build-option=cpu     # Force CPU-only build
pip install -e . --build-option=gpu     # Force GPU build (fallback to CPU if not available)

# Alternative: Direct setup.py commands
python setup.py develop         # Auto-detect (prefer GPU if available)
python setup.py develop cpu     # Force CPU-only build
python setup.py develop gpu     # Force GPU build (fallback to CPU if not available)

# Verify installation
python -c "import sparse_transformers; print('✅ Installation successful')"
```

## Community engagement
We welcome any feedback or suggestions - please join our
[Discord](https://discord.gg/y8WkMncstk) to engage with the community.

## Contributing
We welcome contributions from the community! Areas of particular interest are:
- **Additional models**: Extend beyond LLaMA to other architectures
- **Quantization**: Combine with INT8/FP16 optimizations
- **Attention Kernels**: Implement Sparse Attention Kernels

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

