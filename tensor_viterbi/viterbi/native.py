from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM

try:
    from tensor_viterbi._native import (
        decode_tensor_viterbi_cpp as _decode_cpp,
        decode_tensor_viterbi_cuda as _decode_cuda,
    )
    _NATIVE_AVAILABLE = True
except ImportError as e:
    print(f"[native] ImportError: {e}")
    _NATIVE_AVAILABLE = False


def _arrays(hsmm: HSMM):
    return (
        hsmm.trans_mat,
        hsmm.emission_probs,
        hsmm.start_probs,
        hsmm.duration_probs,
        hsmm.obs_seq.astype(np.int32),
    )


def decode_tensor_viterbi_cpp(hsmm: HSMM) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: pip install -e .")
    return _decode_cpp(*_arrays(hsmm))


def decode_tensor_viterbi_cuda(hsmm: HSMM) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: pip install -e .")
    return _decode_cuda(*_arrays(hsmm))
