from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM

try:
    from ._native import (
        decode_tensor_viterbi_cpp as _decode_cpp,
        decode_tensor_viterbi_cuda as _decode_cuda,
        decode_tensor_viterbi_omp as _decode_omp,
    )
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
    return _decode_omp(n_states, trans_mat, emission_probs, duration_probs_linear, start_probs, duration_probs, obs_seq)
