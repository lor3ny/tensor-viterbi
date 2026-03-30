cmake_minimum_required(VERSION 3.20)

project(tensor-viterbi VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Identify GPU Platform & Hardcode ROCm Paths ---
if(NOT DEFINED GPU_PLATFORM)
    if(DEFINED ENV{ROCM_PATH} OR EXISTS "/opt/rocm-6.3.4")
        set(GPU_PLATFORM "ROCM")
        set(ROCM_PATH "/opt/rocm-6.3.4")
        # Direct CMake to the ROCm-specific modules to avoid "FindHIP" errors
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake/hip" "${ROCM_PATH}")
    else()
        set(GPU_PLATFORM "CUDA")
    endif()
endif()

message(STATUS "Selected GPU Platform: ${GPU_PLATFORM}")

# --- 2. Setup ROCm/HIP Logic ---
if(GPU_PLATFORM STREQUAL "ROCM")
    # Using the config-mode search which is more reliable on LUMI
    find_package(hip CONFIG REQUIRED PATHS ${ROCM_PATH})
    enable_language(HIP)
    
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "GPU Architectures")

    set(CUDA_SOURCES src/hsmm.cu src/kernels.cu)
    set(CUDA_HEADERS src/kernels.cuh)
    set(HIPIFY_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipified_src")
    file(MAKE_DIRECTORY ${HIPIFY_OUT_DIR})

    # Compatibility Layer + Shuffle Performance Fix
    file(WRITE ${HIPIFY_OUT_DIR}/cuda_runtime.h "
#ifndef FAKE_CUDA_RUNTIME_H
#define FAKE_CUDA_RUNTIME_H
#include <hip/hip_runtime.h>
#if !defined(__syncwarp)
    #define __syncwarp() __builtin_amdgcn_wave_barrier()
#endif
#if !defined(__shfl_down_sync)
    #define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta)
#endif
#define hipLaunchCooperativeKernel(f, g, b, p, sh) \\
        (hipLaunchCooperativeKernel)((f), (g), (b), (p), (sh), (hipStream_t)0)
#endif
")

    set(NATIVE_SOURCES "")
    foreach(FILE ${CUDA_SOURCES} ${CUDA_HEADERS})
        get_filename_component(FILENAME ${FILE} NAME)
        set(OUT_FILE "${HIPIFY_OUT_DIR}/${FILENAME}")
        add_custom_command(
            OUTPUT ${OUT_FILE}
            COMMAND ${ROCM_PATH}/bin/hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} > ${OUT_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
            COMMENT "Hipifying: ${FILE}"
        )
        if(${FILE} MATCHES ".*\\.cu$")
            list(APPEND NATIVE_SOURCES ${OUT_FILE})
        endif()
    endforeach()

    set_source_files_properties(${NATIVE_SOURCES} PROPERTIES LANGUAGE HIP)
    add_definitions(-DUSE_ROCM)
else()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "GPU Architectures")
    set(NATIVE_SOURCES src/hsmm.cu src/kernels.cu)
    add_definitions(-DUSE_CUDA)
endif()

# --- 3. Robust Path Detection for Pybind11 ---
execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))" OUTPUT_VARIABLE PYTHON_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import pybind11; print(pybind11.get_include())" OUTPUT_VARIABLE PYBIND_INC OUTPUT_STRIP_TRAILING_WHITESPACE)

# --- 4. Create Extension Module ---
add_library(_native SHARED src/bindings.cpp ${NATIVE_SOURCES})

set_target_properties(_native PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_native"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tensor_viterbi/viterbi
)

# --- 5. Restoring YOUR Optimization Flags ---
target_compile_options(_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-unused-parameter -ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
)

target_include_directories(_native BEFORE PRIVATE ${HIPIFY_OUT_DIR})
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(_native SYSTEM PRIVATE ${PYTHON_INC} ${PYBIND_INC})

if(GPU_PLATFORM STREQUAL "ROCM")
    target_link_libraries(_native PRIVATE hip::host)
endif()
