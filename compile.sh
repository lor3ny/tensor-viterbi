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

# --toolchain all: re-exec for every toolchain defined for this system
if [[ "$TOOLCHAIN" == "all" ]]; then
    _toolchains=$(for k in "${!SYS_MODULES_BUILD[@]}"; do [[ "$k" == "$SYSTEM/"* ]] && echo "${k#*/}"; done | sort)
    if [[ -z "$_toolchains" ]]; then
        echo "Error: No toolchains defined for system '$SYSTEM'."
        exit 1
    fi
    for _tc in $_toolchains; do
        echo "=== Compiling $SYSTEM / $_tc ==="
        "$0" --system "$SYSTEM" --toolchain "$_tc"
    done
    exit $?
fi

if [[ -z "${SYS_MODULES_BUILD[$SYSTEM/$TOOLCHAIN]+x}" ]]; then
    echo "Error: Toolchain '$TOOLCHAIN' is not defined for system '$SYSTEM'."
    _known=$(for k in "${!SYS_MODULES_BUILD[@]}"; do [[ "$k" == "$SYSTEM/"* ]] && echo "  ${k#*/}"; done | sort)
    echo "Known toolchains for $SYSTEM:${_known:- (none)}"
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
IFS=':' read -ra _MODS <<< "$MODULES_BUILD"
for _mod in "${_MODS[@]}"; do
    module load "$_mod"
done

# Create a per-toolchain venv.
# CPU venvs also include hsmmlearn built with the matching compiler.
# GPU venvs only need numpy (run_viterbi.py has no other runtime deps).
# _VENV_CREATED=1 triggers the hsmmlearn build on first setup (CPU only).
_VENV_CREATED=0
VENV_DIR="$SCRIPT_DIR/.venv/$SYSTEM/$TOOLCHAIN"
# Require Python >= 3.10 (numpy 2.x, pybind11 3.x)
_py_ver=$(python3 -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)" 2>/dev/null || echo 0)
if (( _py_ver < 310 )); then
    _py_str=$(python3 --version 2>&1)
    echo "ERROR: Python >= 3.10 required, but found: $_py_str"
    echo "       Load a newer Python module for $SYSTEM/$TOOLCHAIN in systems.conf"
    exit 1
fi

if [[ ! -f "$VENV_DIR/bin/python3" ]]; then
    echo "Creating venv at $VENV_DIR ..."
    if [[ "$TYPE" == "gpu" ]]; then
        # Inherit system-wide site-packages so numpy and pybind11 installed
        # by the system Python module are visible without a separate pip install.
        python3 -m venv --system-site-packages "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
        _VENV_CREATED=1
    fi
fi
PYTHON_EXE="$VENV_DIR/bin/python3"

if [[ "$TYPE" == "gpu" ]]; then
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
    CMAKE_FLAGS=(-DBUILD_GPU=OFF -DSYSTEM_NAME="$CMAKE_SYSTEM_NAME")
fi

# Build the native extension (always rebuilds; remove build dir is not needed).
rm -rf "$BUILD_DIR"
cmake -B "$BUILD_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXE" "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD_DIR" -j 8

# Build hsmmlearn packages with the active toolchain so the OMP runtime
# (libgomp vs libcraymp vs libiomp5) matches _native.so, and vectorization is
# on equal footing between baseline and tensor implementations.
# GPU systems do not run these baselines, so skip them there.
# Only build when the venv was just created; remove the venv dir to force a rebuild.
if [[ "$TYPE" == "cpu" && $_VENV_CREATED -eq 1 ]]; then
    case "$TOOLCHAIN" in
        cray) _CC=cc;    _CXX=CC     ;;
        intel)  _CC=icx;   _CXX=icpx   ;;
        llvm)   _CC=clang; _CXX=clang++ ;;
        *)      _CC=gcc;   _CXX=g++    ;;
    esac
    echo "Building hsmmlearn packages with CC=$_CC CXX=$_CXX ..."
    # wheel must be present so setuptools can build the legacy hsmmlearn_omp package
    "$VENV_DIR/bin/pip" install --quiet wheel
    CC="$_CC" CXX="$_CXX" "$PYTHON_EXE" -m pip install --no-build-isolation "$SCRIPT_DIR/hsmmlearn"
    CC="$_CC" CXX="$_CXX" "$PYTHON_EXE" -m pip install --no-build-isolation "$SCRIPT_DIR/hsmmlearn_omp"
fi
