# LLaMA Skip Connection Optimization

A PyTorch C++ implementation of optimized sparse MLP operations for LLaMA models with skip connections.

## Overview

This project implements and optimizes sparse MLP operations in LLaMA models using:
- Custom C++ CUDA kernels
- Efficient weight selection
- Parallel processing optimizations
- Memory layout optimizations

## Installation

1. Requirements:
```bash
torch>=1.10.0
transformers>=4.20.0
ninja
```

2. Build C++ extensions:
```bash
# Install in editable mode
pip install -e .
```

3. Verify installation:
```bash
python -c "import sparse_mlp; print('Installation successful!')"
```

## Usage

### Running Benchmarks

```bash
# Run on CUDA with config and verbose output
python trainer_notebook.py \
    --device cuda \
    --config configs/llama_skip_causal_3b.json \
    --num_runs 50 \
    --verbose True

# Run on CPU with timing details
python trainer_notebook.py \
    --device cpu \
    --config configs/llama_skip_causal_3b.json \
    --verbose True
```

Available arguments:
- `--device`: Device to run on (`cuda` or `cpu`, default: `cpu`)
- `--config`: Path to model config file (default: `configs/llama_skip_causal_3b.json`)
- `--num_runs`: Number of inference runs (default: 50)
- `--verbose`: Enable detailed timing output (default: False)

### Latest Performance Results

#### CUDA Performance
```
Standard LLaMA:
- Average time: 0.018s
- Min: 0.017s
- Max: 0.018s

SkipLLaMA Scripted:
- Average time: 0.021s
- Min: 0.021s
- Max: 0.021s

Current CUDA speedup: 0.85x (optimization in progress)
```

#### CPU Performance
```
Standard LLaMA:
- Average time: 1.824s
- Min: 1.095s
- Max: 6.964s

SkipLLaMA Scripted:
- Average time: 1.095s
- Min: 0.741s
- Max: 4.726s

CPU speedup: 1.67x
```

## Implementation Details

### Core Components

1. Sparse MLP Operation (`sparse_mlp/csrc/sparse_mlp_op.cpp`):
```cpp
torch::Tensor sparse_mlp_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor mask,
    std::string act_fn_name)
```

2. Skip Connection Model (`src/models/modelling_llama_skip.py`):
```python
class LlamaSkipConnectionForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaSkipModel(config)
```

## Project Structure

```
├── sparse_mlp/
│   └── csrc/
│       ├── sparse_mlp_op.cpp    # C++ implementation
│       ├── sparse_mlp_cuda.cu   # CUDA kernels
│       ├── weight_cache.h       # Weight caching
│       └── timer.h             # Timing utilities
├── src/
│   └── models/
│       ├── modelling_llama_skip.py      # PyTorch model
│       └── configuration_llama_skip.py   # Model config
├── configs/
│   ├── llama_skip_causal_1b.json        # 1B model config
│   └── llama_skip_causal_3b.json        # 3B model config
├── setup.py                     # Build script
└── README.md
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Open pull request

## Citation

If you use this code, please cite:
```bibtex
@software{llama_skip_connection,
  title = {LLaMA Skip Connection Optimization},
  year = {2024},
  author = {Author},
  url = {https://github.com/username/llama-skip}
}
```