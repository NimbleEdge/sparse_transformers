import os
import torch
# Configure CPU threads
num_threads = os.cpu_count()
print(f"Configuring for {num_threads} CPU threads")
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
os.environ['MAX_JOBS'] = str(num_threads)

# torch.ops.load_library("./build/lib/sparse_mlp.so")
torch.classes.load_library("./build/lib/sparse_mlp.so")

sparse_mlp_forward = torch.ops.sparse_mlp.forward
WeightCache = torch.classes.sparse_mlp.WeightCache


# Re-export for API compatibility
__all__ = [
    'sparse_mlp_forward', 
    'WeightCache'
]