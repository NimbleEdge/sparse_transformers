from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os
import torch
from pathlib import Path
import shutil
import sys

# Create build directory if it doesn't exist
build_dir = Path(__file__).parent / 'build'
if build_dir.exists():
    shutil.rmtree(build_dir)
build_dir.mkdir(parents=True)
(build_dir / 'lib').mkdir(exist_ok=True)

# Set environment variables to control build output
os.environ['TORCH_BUILD_DIR'] = str(build_dir)
os.environ['BUILD_LIB'] = str(build_dir / 'lib')
os.environ['BUILD_TEMP'] = str(build_dir / 'temp')

# Get CUDA compute capability if GPU is available
arch_flags = []
if torch.cuda.is_available():
    arch_list = []
    for i in range(torch.cuda.device_count()):
        arch_list.append(torch.cuda.get_device_capability(i))
    arch_list = sorted(list(set(arch_list)))
    arch_flags = [f"-gencode=arch=compute_{arch[0]}{arch[1]},code=sm_{arch[0]}{arch[1]}" for arch in arch_list]

# Common optimization flags
common_compile_args = [
    '-O3',                      # Maximum optimization
    '-march=native',            # Optimize for local CPU architecture
    '-ffast-math',              # Aggressive floating point optimizations
    '-fopenmp',                 # OpenMP support
    '-flto',                    # Link-time optimization
    '-funroll-loops',           # Unroll loops
    '-fno-math-errno',          # Assume math functions never set errno
    '-fno-trapping-math',       # Assume FP ops don't generate traps
    '-mtune=native',            # Tune code for local CPU
]

# CPU-specific optimization flags
cpu_compile_args = common_compile_args + [
    '-mavx2',                   # Enable AVX2 instructions if available
    '-mfma',                    # Enable FMA instructions if available
    '-fno-plt',                 # Improve indirect call performance
    '-fuse-linker-plugin',      # Enable LTO plugin
    '-fomit-frame-pointer',     # Remove frame pointers
    '-fno-stack-protector',     # Disable stack protector
    '-fvisibility=hidden',      # Hide all symbols by default
    '-fdata-sections',          # Place each data item into its own section
    '-ffunction-sections',      # Place each function into its own section
]

# CUDA-specific optimization flags
cuda_compile_args = ['-O3', '--use_fast_math'] + arch_flags + [
    '--compiler-options', "'-fPIC'",
    '--compiler-options', "'-O3'",
    '--compiler-options', "'-march=native'",
    '--compiler-options', "'-ffast-math'",
    '-std=c++17'  # Force C++17 instead of C++20
]

# Link flags
extra_link_args = [
    '-fopenmp',
    '-flto',                    # Link-time optimization
    '-fuse-linker-plugin',      # Enable LTO plugin
    '-Wl,--as-needed',          # Only link needed libraries
    '-Wl,-O3',                  # Linker optimizations
    '-Wl,--strip-all',          # Strip all symbols
    '-Wl,--gc-sections',        # Remove unused sections
    '-Wl,--exclude-libs,ALL',   # Don't export any symbols from libraries
]

# Get CUDA include paths
def get_cuda_include_dirs():
    cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(cuda_home):
        cuda_home = os.getenv('CUDA_PATH')  # Windows
    
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME or CUDA_PATH environment variable is not set")
        
    return [
        os.path.join(cuda_home, 'include'),
        os.path.join(cuda_home, 'samples', 'common', 'inc')
    ]

# Base extension configuration
base_include_dirs = [
    os.path.dirname(torch.__file__) + '/include',
    os.path.dirname(torch.__file__) + '/include/torch/csrc/api/include',
    os.path.dirname(torch.__file__) + '/include/ATen',
    os.path.dirname(torch.__file__) + '/include/c10',
]

if torch.cuda.is_available():
    base_include_dirs.extend(get_cuda_include_dirs())

# Define extensions
ext_modules = []
if torch.cuda.is_available():
    extension = CUDAExtension(
        name='sparse_mlp',
        sources=[
            'sparse_mlp/csrc/sparse_mlp_op.cpp',
            'sparse_mlp/csrc/sparse_mlp_cuda.cu'
        ],
        include_dirs=base_include_dirs,
        extra_compile_args={
            'cxx': cpu_compile_args,
            'nvcc': cuda_compile_args
        },
        extra_link_args=extra_link_args,
        libraries=['gomp', 'cudart'],
        library_dirs=[str(build_dir / 'lib')],
        define_macros=[('WITH_CUDA', None)]
    )
else:
    extension = CppExtension(
        name='sparse_mlp',
        sources=['sparse_mlp/csrc/sparse_mlp_op.cpp'],
        extra_compile_args=cpu_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=[str(build_dir / 'lib')],
        include_dirs=base_include_dirs,
        libraries=['gomp']
    )

ext_modules.append(extension)

# Custom build extension to handle clean builds
class CustomBuildExtension(BuildExtension):
    def get_ext_filename(self, ext_name):
        # Force output to build directory
        filename = super().get_ext_filename(ext_name)
        return str(build_dir / 'lib' / os.path.basename(filename))

    def get_ext_fullpath(self, ext_name):
        # Override to ensure extension is built in our build directory
        filename = self.get_ext_filename(ext_name)
        return str(build_dir / 'lib' / filename)
    
    def build_extensions(self):
        # Clean old build files
        if self.parallel:
            self.parallel = False  # Disable parallel build for CUDA
        super().build_extensions()

setup(
    name='sparse_mlp',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExtension.with_options(no_python_abi_suffix=True),
    }
) 