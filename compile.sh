#!/bin/bash
# Usage: ./compile.sh --system <system_name>
# Available systems are defined in systems.conf
# Builds either the CPU (OpenMP) or GPU (CUDA/ROCm) backend — never both.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$PWD" != "$SCRIPT_DIR" ]]; then
    echo "Error: must be run from the script's directory."
    echo "  cd \"$SCRIPT_DIR\" && $0 $*"
    exit 1
fi
source "$SCRIPT_DIR/systems.conf"

SYSTEM=""
TOOLCHAIN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system|-s)    SYSTEM="$2";    shift 2 ;;
        --toolchain|-t) TOOLCHAIN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SYSTEM" ]]; then
    echo "Error: --system argument is required."
    echo "Available systems: ${!SYS_TYPE[*]}"
    exit 1
fi

if [[ -z "$TOOLCHAIN" ]]; then
    echo "Error: --toolchain argument is required."
    exit 1
fi

if [[ -z "${SYS_TYPE[$SYSTEM]+x}" ]]; then
    echo "Error: Unknown system '$SYSTEM'."
    echo "Available systems: ${!SYS_TYPE[*]}"
    exit 1
fi

TYPE="${SYS_TYPE[$SYSTEM]}"
PARTITION="${SYS_PARTITION[$SYSTEM]}"
ACCOUNT="${SYS_ACCOUNT[$SYSTEM]}"
GPU_ARCH="${SYS_GPU_ARCH[$SYSTEM]}"

MODULES_BUILD="${SYS_MODULES_BUILD[$SYSTEM/$TOOLCHAIN]}"
BUILD_DIR="build/$SYSTEM/$TOOLCHAIN"
CMAKE_SYSTEM_NAME="$SYSTEM/$TOOLCHAIN"

# Load build modules
#module purge
PYTHON_EXE="$(command -v python3)"  # capture venv python BEFORE module load overrides PATH
IFS=':' read -ra _MODS <<< "$MODULES_BUILD"
for _mod in "${_MODS[@]}"; do
    module load "$_mod"
done

rm -rf "$BUILD_DIR"

if [[ "$TYPE" == "gpu" ]]; then
    SRUN_FLAGS=(--gres=gpu:1 -A "$ACCOUNT" -p "$PARTITION")

    if module list 2>&1 | grep -qi rocm; then
        CMAKE_FLAGS=(
            -DBUILD_GPU=ON
            -DGPU_PLATFORM=ROCM
            -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH"
            -DSYSTEM_NAME="$CMAKE_SYSTEM_NAME"
        )
    else
        CMAKE_FLAGS=(
            -DBUILD_GPU=ON
            -DGPU_PLATFORM=CUDA
            -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH"
            -DSYSTEM_NAME="$CMAKE_SYSTEM_NAME"
        )
    fi
else
    SRUN_FLAGS=(-A "$ACCOUNT" -p "$PARTITION")
    CMAKE_FLAGS=(-DBUILD_GPU=OFF -DSYSTEM_NAME="$CMAKE_SYSTEM_NAME")
fi

srun "${SRUN_FLAGS[@]}" cmake -B "$BUILD_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXE" "${CMAKE_FLAGS[@]}"
srun "${SRUN_FLAGS[@]}" cmake --build "$BUILD_DIR" -j 8
