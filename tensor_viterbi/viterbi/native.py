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
if _SYSTEM:
    _so_dir = str(_HERE / _SYSTEM)
    if _so_dir not in _sys.path:
        _sys.path.insert(0, _so_dir)

try:
    if _SYSTEM:
        import importlib as _importlib
        _native = _importlib.import_module("_native")
        _decode_cpp  = _native.decode_tensor_viterbi_cpp
        _decode_cuda = getattr(_native, "decode_tensor_viterbi_cuda", None)
        _decode_omp  = getattr(_native, "decode_tensor_viterbi_omp",  None)
    else:
        from ._native import (
            decode_tensor_viterbi_cpp  as _decode_cpp,
        )
        try:
            from ._native import decode_tensor_viterbi_cuda as _decode_cuda
        except ImportError:
            _decode_cuda = None
        try:
            from ._native import decode_tensor_viterbi_omp as _decode_omp
        except ImportError:
            _decode_omp = None
    _NATIVE_AVAILABLE = True
except ImportError as e:
    print(f"[native] ImportError: {e}")
    _NATIVE_AVAILABLE = False




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
