import os
import torch

# Configure CPU threads and threading
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

# Enable TorchScript optimizations
torch.jit.enable_onednn_fusion(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

torch.classes.load_library(os.path.join(os.path.dirname(__file__), "sparse_transformers.so"))

# Only define these if the extension loaded successfully
sparse_mlp_forward = torch.ops.sparse_mlp.forward
sparse_mlp_forward_opt = torch.ops.sparse_mlp.forward_opt
WeightCache = torch.classes.sparse_mlp.WeightCache
WeightCacheOpt = torch.classes.sparse_mlp.WeightCacheOpt
approx_topk_threshold = torch.ops.sparse_mlp.approx_topk_threshold
__all__ = [
    'sparse_mlp_forward',
    'sparse_mlp_forward_opt',
    'WeightCache',
    'WeightCacheOpt',
    'approx_topk_threshold'
]