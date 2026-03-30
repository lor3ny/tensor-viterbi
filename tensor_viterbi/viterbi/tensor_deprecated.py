from tensor_viterbi.hsmm import HSMM
from .tensor import _backtracking, _compute_survival_probs, _tail_adjustment
import numpy as np


def decode_log_tensor_viterbi_no_cache(hsmm: HSMM) -> np.ndarray:
    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), -np.inf)
    delta_state = np.zeros((T, N), dtype=int)
    delta_dur = np.zeros((T, N), dtype=int)

    survival_probs = _compute_survival_probs(hsmm.duration_probs)

    #! PHASE 1 - INITIALIZATION 0<=t<D
    PAST_DELTA = hsmm.duration_probs + hsmm.start_probs[np.newaxis, :]

    obs_indices = hsmm.obs_seq[:D].astype(int)
    emission_rows = hsmm.emission_probs[obs_indices, :]
    cum_emission = np.cumsum(emission_rows, axis=0)
    EMISSION_PROBS = cum_emission

    delta[0:D, :] = PAST_DELTA + EMISSION_PROBS
    delta_dur[0:D] = np.arange(1, D + 1)[:, np.newaxis] * np.ones((1, N), dtype=int)

    #! PHASE 2 - INDUCTION  t>0
    AP = hsmm.trans_mat[np.newaxis, :, :] + hsmm.duration_probs[:, :, np.newaxis]
    for t in range(1, T):
        segment_indices = hsmm.obs_seq[max(0, t - D + 1) : t + 1].astype(int)
        relevant_probs = hsmm.emission_probs[segment_indices, :]
        cum_emission = np.cumsum(np.flip(relevant_probs, axis=0), axis=0)
        EMISSION_PROBS[: cum_emission.shape[0], :] = cum_emission

        window = delta[max(0, t - D) : t, :]
        PAST_DELTA[: window.shape[0], :] = window[::-1]

        DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] + AP
        RESULT_B = EMISSION_PROBS[:, :, np.newaxis] + DELTA_EMISSION

        planes = RESULT_B.transpose(1, 0, 2)
        sliced = planes[:, : min(t, D), :]
        flat_idx = np.argmax(sliced.reshape(N, -1), axis=1)

        slice_shape = sliced.shape[1:]
        d_arr, i_arr = np.unravel_index(flat_idx, slice_shape)

        best_vals = planes[np.arange(N), d_arr, i_arr]

        if t < D:
            cond = best_vals < delta[t, :]
        else:
            cond = np.zeros(N, dtype=bool)

        delta[t, :] = np.where(cond, delta[t, :], best_vals)
        delta_state[t, :] = np.where(cond, delta_state[t, :], i_arr)
        delta_dur[t, :] = np.where(cond, delta_dur[t, :], d_arr + 1)

    _tail_adjustment(delta, delta_state, delta_dur,
                     EMISSION_PROBS, PAST_DELTA, survival_probs,
                     hsmm.trans_mat, T, N, D)

    return _backtracking(delta, delta_state, delta_dur, T)
