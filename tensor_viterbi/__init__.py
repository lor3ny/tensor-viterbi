from .hsmm import HSMM
from .viterbi import (
    decode_tensor_viterbi,
    decode_log_tensor_viterbi_no_cache,
    decode_log_tensor_viterbi_cached,
    decode_vanilla_viterbi,
)

try:
    from .viterbi._native import decode_tensor_viterbi_cpp, decode_tensor_viterbi_cuda
except ImportError:
    decode_tensor_viterbi_cpp = None
    decode_tensor_viterbi_cuda = None

__all__ = [
    "HSMM",
    "decode_tensor_viterbi",
    "decode_log_tensor_viterbi_no_cache",
    "decode_log_tensor_viterbi_cached",
    "decode_vanilla_viterbi",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_cuda",
]
