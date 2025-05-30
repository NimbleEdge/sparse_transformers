cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(sparse_mlp)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Define our library target
add_library(sparse_mlp SHARED 
    csrc/sparse_mlp_op.cpp
)

# Enable C++17
target_compile_features(sparse_mlp PRIVATE cxx_std_17)

# Add OpenMP flags
target_compile_options(sparse_mlp PRIVATE ${OpenMP_CXX_FLAGS})

# Include directories
target_include_directories(sparse_mlp PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

# Link against LibTorch and OpenMP
target_link_libraries(sparse_mlp PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Set output directory
set_target_properties(sparse_mlp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

# Add optimization flags
target_compile_options(sparse_mlp PRIVATE
    -O3
    -ffast-math
    -march=native
) 