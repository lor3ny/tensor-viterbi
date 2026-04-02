from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM

import os as _os
import sys as _sys
import pathlib as _pathlib

_HERE = _pathlib.Path(__file__).parent.resolve()
_SYSTEM = _os.environ.get("SYS_NAME", "")
if not _SYSTEM:
    raise RuntimeError(
        "[native] SYS_NAME is not set.\n"
        "Pass --system <system/toolchain> to run_viterbi.py, "
        "or set SYS_NAME in the environment."
    )

_so_dir = str(_HERE / _SYSTEM)
if _so_dir not in _sys.path:
    _sys.path.insert(0, _so_dir)

try:
    import importlib as _importlib
    _native = _importlib.import_module("_native")
    _decode_cpp  = _native.decode_tensor_viterbi_cpp
    _decode_cuda = getattr(_native, "decode_tensor_viterbi_cuda", None)
    _decode_omp  = getattr(_native, "decode_tensor_viterbi_omp",  None)
    _NATIVE_AVAILABLE = True
except ImportError as e:
    raise RuntimeError(
        f"[native] Could not load _native extension from '{_so_dir}'.\n"
        f"Run: compile.sh --system <system> --toolchain <toolchain>\n"
        f"Original error: {e}"
    ) from e




def decode_tensor_viterbi_cpp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq
) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: cmake -B build && cmake --build build")
    return _decode_cpp(n_states, trans_mat, emission_probs, duration_probs_linear, start_probs, duration_probs, obs_seq)


def decode_tensor_viterbi_cuda(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq
) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: cmake -B build && cmake --build build")
    if _decode_cuda is None:
        raise RuntimeError("CUDA backend not available in this build (compiled with NO_GPU)")
    return _decode_cuda(n_states, trans_mat, emission_probs, duration_probs_linear, start_probs, duration_probs, obs_seq)


def decode_tensor_viterbi_omp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq
) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: cmake -B build && cmake --build build")
    if _decode_omp is None:
        raise RuntimeError("OMP backend not available in this build")
    return _decode_omp(n_states, trans_mat, emission_probs, duration_probs_linear, start_probs, duration_probs, obs_seq)
