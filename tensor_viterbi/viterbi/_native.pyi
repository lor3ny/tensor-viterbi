import numpy as np

def decode_tensor_viterbi_cpp(
    trans_mat: np.ndarray,
    emission_probs: np.ndarray,
    start_probs: np.ndarray,
    duration_probs: np.ndarray,
    obs_seq: np.ndarray,
) -> np.ndarray: ...

def decode_tensor_viterbi_cuda(
    trans_mat: np.ndarray,
    emission_probs: np.ndarray,
    start_probs: np.ndarray,
    duration_probs: np.ndarray,
    obs_seq: np.ndarray,
) -> np.ndarray: ...
