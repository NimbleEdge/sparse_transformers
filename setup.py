from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os
import torch
from pathlib import Path
import shutil
import sys
import warnings

# Check PyTorch C++ ABI compatibility
def get_pytorch_abi_flag():
    """Get the correct C++ ABI flag to match PyTorch compilation."""
    return f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}'

# Get PyTorch ABI flag
pytorch_abi_flag = get_pytorch_abi_flag()
print(f"Using PyTorch C++ ABI flag: {pytorch_abi_flag}")

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
    try:
        arch_list = []
        for i in range(torch.cuda.device_count()):
            arch_list.append(torch.cuda.get_device_capability(i))
        arch_list = sorted(list(set(arch_list)))
        arch_flags = [f"-gencode=arch=compute_{arch[0]}{arch[1]},code=sm_{arch[0]}{arch[1]}" for arch in arch_list]
        print(f"CUDA architectures detected: {arch_list}")
    except Exception as e:
        warnings.warn(f"Error detecting CUDA architecture: {e}")
        # Use a common architecture as fallback
        arch_flags = ["-gencode=arch=compute_86,code=sm_86"]

# Common optimization flags (compatible with both old and new ABI)
common_compile_args = [
    '-O3',                      # Maximum optimization
    '-fopenmp',                 # OpenMP support
    '-flto',                    # Link-time optimization
    '-funroll-loops',           # Unroll loops
    '-fno-math-errno',          # Assume math functions never set errno
    '-fno-trapping-math',       # Assume FP ops don't generate traps
    '-mtune=native',            # Tune code for local CPU
    pytorch_abi_flag,           # Critical: Match PyTorch's C++ ABI
    '-DTORCH_API_INCLUDE_EXTENSION_H',  # PyTorch extension header compatibility
]

# Try to detect if we can use advanced CPU optimizations safely
try:
    import platform
    if platform.machine() in ['x86_64', 'AMD64']:
        advanced_cpu_flags = [
            '-march=native',            # Optimize for local CPU architecture
            '-mtune=native',            # Tune code for local CPU
            '-mavx2',                   # Enable AVX2 instructions if available
            '-mfma',                    # Enable FMA instructions if available
        ]
    else:
        advanced_cpu_flags = []
except:
    advanced_cpu_flags = []

# CPU-specific optimization flags
cpu_compile_args = common_compile_args + advanced_cpu_flags + [
    '-flto',                    # Link-time optimization
    '-funroll-loops',           # Unroll loops
    '-fno-math-errno',          # Assume math functions never set errno
    '-fno-trapping-math',       # Assume FP ops don't generate traps
    '-fno-plt',                 # Improve indirect call performance
    '-fuse-linker-plugin',      # Enable LTO plugin
    '-fomit-frame-pointer',     # Remove frame pointers
    '-fno-stack-protector',     # Disable stack protector
    '-fvisibility=hidden',      # Hide all symbols by default
    '-fdata-sections',          # Place each data item into its own section
    '-ffunction-sections',      # Place each function into its own section
    '-fvisibility=default',
]

# CUDA-specific optimization flags (ensure C++17 compatibility and ABI matching)
cuda_compile_args = ['-O3', '--use_fast_math'] + arch_flags + [
    '--compiler-options', f"'-fPIC'",
    '--compiler-options', f"'-O3'",
    '-std=c++17',               # Force C++17 for compatibility
    '--compiler-options', "'-fvisibility=default'",
]

# Add advanced CPU flags to CUDA compilation if available
if advanced_cpu_flags:
    for flag in ['-march=native', '-ffast-math']:
        cuda_compile_args.extend(['--compiler-options', f"'{flag}'"])

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
        # Try common CUDA locations
        for path in ['/usr/local/cuda', '/opt/cuda', '/usr/cuda']:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if cuda_home is None:
        warnings.warn("CUDA installation not found. CUDA extensions will not be built.")
        return []
        
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

# Define extensions
ext_modules = []

cpp_source = 'sparse_transformers/csrc/sparse_mlp_op.cpp'
cuda_source = 'sparse_transformers/csrc/sparse_mlp_cuda.cu'

if not os.path.exists(cpp_source):
    warnings.warn(f"C++ source file not found: {cpp_source}")
    raise FileNotFoundError(f"Missing source file: {cpp_source}")

if torch.cuda.is_available() and os.path.exists(cuda_source):
    print("Building CUDA extension...")
    cuda_include_dirs = get_cuda_include_dirs()
    if cuda_include_dirs:
        base_include_dirs.extend(cuda_include_dirs)
        extension = CUDAExtension(
            name='sparse_transformers.sparse_transformers',
            sources=[cpp_source, cuda_source],
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
        print("CUDA include directories not found, falling back to CPU-only extension...")
        raise RuntimeError("CUDA headers not found")
else:
    print("Building CPU-only extension...")
    cuda_include_dirs = get_cuda_include_dirs()
    if cuda_include_dirs:
        base_include_dirs.extend(cuda_include_dirs)
    extension = CppExtension(
        name='sparse_transformers.sparse_transformers',
        sources=[cpp_source],
        extra_compile_args=cpu_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=[str(build_dir / 'lib')],
        include_dirs=base_include_dirs,
        libraries=['gomp']
    )

ext_modules.append(extension)
print(f"Extension configured successfully: {extension.name}")


# Custom build extension to handle clean builds and ABI compatibility
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
        # Disable parallel build for better error reporting and CUDA compatibility
        if self.parallel:
            self.parallel = False
            
        # Print compilation info for debugging
        print(f"Building extensions with PyTorch {torch.__version__}")
        print(f"PyTorch C++ ABI: {pytorch_abi_flag}")
        super().build_extensions()
        print("C++ extension built successfully!")

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / 'requirements.txt'
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name='sparse_transformers',
    version='0.0.1',
    description='Sparse Inferencing for transformer based LLMs',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExtension.with_options(no_python_abi_suffix=True),
    },
    install_requires=read_requirements(),
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,  # Required for C++ extensions
) 