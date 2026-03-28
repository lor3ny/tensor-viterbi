cmake_minimum_required(VERSION 3.18)
project(tensor_viterbi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── pybind11 ──────────────────────────────────────────────────────────────── #
# 1. Run the python command to find the pybind11 CMake directory
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FIND_RESULT
)

# 2. Check if the command succeeded
if(NOT PYBIND11_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find pybind11 via python. Is it installed?")
endif()

# 3. Tell CMake to look in that specific directory
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

# Force pybind11 to use the modern FindPython (prevents CMP0148 warning)
set(PYBIND11_FINDPYTHON ON)

# 4. Now find_package will work!
find_package(pybind11 CONFIG REQUIRED)

# ── CUDA architecture ───────────────────────────── #
set(CMAKE_CUDA_ARCHITECTURES native)

# ── Native extension: _native.so ─────────────────────────────────────────── #
pybind11_add_module(_native
    src/bindings.cpp
    src/hsmm.cu
    src/kernels.cu
)

target_include_directories(_native PRIVATE src/src)

target_compile_options(_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
)

# Drop the .so directly into the Python package — no install step needed
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tensor_viterbi/viterbi
)
