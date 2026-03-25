cmake_minimum_required(VERSION 3.18)
project(tensor_viterbi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── pybind11 ──────────────────────────────────────────────────────────────── #
find_package(pybind11 CONFIG REQUIRED)

# ── CUDA architecture (A100 @ CINECA Leonardo) ───────────────────────────── #
set(CMAKE_CUDA_ARCHITECTURES 80)

# ── Native extension: _native.so ─────────────────────────────────────────── #
pybind11_add_module(_native
    src/bindings.cpp
    src/src/hsmm.cu
    src/src/kernels.cu
)

target_include_directories(_native PRIVATE src/src)

target_compile_options(_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler "-Wall -Wextra -Wno-unused-parameter">
)

# Install the extension inside the Python package
install(TARGETS _native DESTINATION tensor_viterbi/viterbi)
