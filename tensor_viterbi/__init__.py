from .hsmm import HSMM
from .viterbi import (
    decode_log_tensor_viterbi_no_cache,
    decode_log_tensor_viterbi_cached,
    decode_vanilla_viterbi
)

try:
    from .viterbi.native import decode_tensor_viterbi_cpp, decode_tensor_viterbi_cuda, decode_tensor_viterbi_omp
except ImportError:
    decode_tensor_viterbi_cpp = None
    decode_tensor_viterbi_cuda = None
    decode_tensor_viterbi_omp = None

__all__ = [
    "HSMM",
    "decode_log_tensor_viterbi_no_cache",
    "decode_log_tensor_viterbi_cached",
    "decode_vanilla_viterbi",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_cuda",
    "decode_tensor_viterbi_omp",
]
