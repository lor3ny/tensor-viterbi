from typing import List
import numpy as np

def decode_tensor_viterbi_cpp(
    states: int,
    trans_mat: np.ndarray,
    emission_probs: np.ndarray,
    emission_probs_linear: np.ndarray,
    start_probs: np.ndarray,
    duration_probs: np.ndarray,
    obs_seq: np.ndarray,
) -> np.ndarray: ...

def decode_tensor_viterbi_cuda(
    states: int,
    trans_mat: np.ndarray,
    emission_probs: np.ndarray,
    emission_probs_linear: np.ndarray,
    start_probs: np.ndarray,
    duration_probs: np.ndarray,
    obs_seq: np.ndarray,
) -> np.ndarray: ...
