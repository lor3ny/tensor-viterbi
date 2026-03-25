from __future__ import annotations

import numpy as np

try:
    from ._native import (
        decode_tensor_viterbi_cpp as _decode_cpp,
        decode_tensor_viterbi_cuda as _decode_cuda,
    )
    _NATIVE_AVAILABLE = True
except ImportError as e:
    print(f"[native] ImportError: {e}")
    _NATIVE_AVAILABLE = False


def decode_tensor_viterbi_cpp(json_path: str) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: cmake -B build && cmake --build build")
    return _decode_cpp(json_path)


def decode_tensor_viterbi_cuda(json_path: str) -> np.ndarray:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Native extension not built. Run: cmake -B build && cmake --build build")
    return _decode_cuda(json_path)
