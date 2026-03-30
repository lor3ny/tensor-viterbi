from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM


def _backtracking(
        delta: np.ndarray, 
        psi_state: np.ndarray, 
        psi_dur: np.ndarray, 
        T: int
) -> np.ndarray:
    
    path = np.zeros(T, dtype=int)

    t = T - 1
    best_last_state = np.argmax(delta[t])
    curr_state = best_last_state

    while t >= 0:
        d = psi_dur[t, curr_state]
        prev_s = psi_state[t, curr_state]

        start_t = t - d + 1
        path[start_t : t + 1] = curr_state

        t = t - d
        curr_state = prev_s

    return path


def _compute_survival_probs(
        duration_probs: np.ndarray
) -> np.ndarray:

    D, N = duration_probs.shape
    survival_probs = np.full((D, N), -np.inf)
    survival_probs[-1] = duration_probs[-1]
    for d in range(D - 2, -1, -1):
        m = np.maximum(survival_probs[d + 1], duration_probs[d])
        finite = m > -np.inf
        survival_probs[d] = np.where(
            finite,
            m + np.log(np.exp(survival_probs[d + 1] - m) + np.exp(duration_probs[d] - m)),
            -np.inf,
        )
    return survival_probs


def _tail_adjustment(
    delta: np.ndarray,
    psi_state: np.ndarray,
    psi_dur: np.ndarray,
    EMISSION_PROBS: np.ndarray,
    PAST_DELTA: np.ndarray,
    survival_probs: np.ndarray,
    trans_mat: np.ndarray,
    T: int,
    N: int,
    D: int,
) -> None:

    tau = min(T - 1, D)

    AP_tail = trans_mat[np.newaxis, :, :] + survival_probs[:, :, np.newaxis]    # (D, N, N)

    DELTA_EMISSION = PAST_DELTA[:tau, np.newaxis, :] + AP_tail[:tau]            # (tau, N, N)
    RESULT         = EMISSION_PROBS[:tau, :, np.newaxis] + DELTA_EMISSION       # (tau, N, N)

    planes   = RESULT.transpose(1, 0, 2)                                        # (N, tau, N)
    flat_idx = np.argmax(planes.reshape(N, -1), axis=1)                         # (N,)
    d_arr, i_arr = np.unravel_index(flat_idx, (tau, N))

    delta[T - 1, :]       = planes[np.arange(N), d_arr, i_arr]
    psi_state[T - 1, :] = i_arr
    psi_dur[T - 1, :]   = d_arr + 1



#? We use the official signature of Numpy for 3D tensors (d,y,x)
#? axis 0 → depth (z) — the first index, selects a 2D "slice"
#? axis 1 → rows — the second index, selects a row within a slice
#? axis 2 → columns — the third index, selects a column within a row
def decode_log_tensor_viterbi_cached(
        hsmm: HSMM
) -> np.ndarray:
    
    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), -np.inf)
    psi_state = np.zeros((T, N), dtype=int)
    psi_dur = np.zeros((T, N), dtype=int)

    #! PHASE 1 - INITIALIZATION 0<=t<D
    survival_probs = _compute_survival_probs(hsmm.duration_probs)

    PAST_DELTA = hsmm.duration_probs + hsmm.start_probs[np.newaxis, :]

    obs_indices = hsmm.obs_seq[:D].astype(int)
    emission_rows = hsmm.emission_probs[obs_indices, :]
    cum_emission = np.cumsum(emission_rows, axis=0)
    EMISSION_PROBS = cum_emission

    delta[0:D] = PAST_DELTA + EMISSION_PROBS
    psi_dur[0:D] = np.arange(1, D + 1)[:, np.newaxis] * np.ones((1, N), dtype=int)

    #! PHASE 2 - INDUCTION  t>0
    AP = hsmm.trans_mat[np.newaxis, :, :] + hsmm.duration_probs[:, :, np.newaxis]
    EMISSION_CACHE = np.zeros((D, N), dtype=float)
    for t in range(1, T):
        if t > D:
            _index_t = hsmm.obs_seq[t].astype(int)
            _probs_t = hsmm.emission_probs[_index_t, :]

            EMISSION_PROBS = EMISSION_CACHE + _probs_t

            EMISSION_CACHE[1:, :] = EMISSION_PROBS[: D - 1, :]
        else:
            segment_indices = hsmm.obs_seq[max(0, t - D + 1) : t + 1].astype(int)
            relevant_probs = hsmm.emission_probs[segment_indices, :]
            cum_emission = np.cumsum(np.flip(relevant_probs, axis=0), axis=0)

            EMISSION_PROBS[: cum_emission.shape[0], :] = cum_emission

            if t == D:
                EMISSION_CACHE[1:, :] = cum_emission[: D - 1, :]

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
        psi_state[t, :] = np.where(cond, psi_state[t, :], i_arr)
        psi_dur[t, :] = np.where(cond, psi_dur[t, :], d_arr + 1)

    #! TAIL ADJUSTMENT — t = T-1
    _tail_adjustment(delta, psi_state, psi_dur,
                     EMISSION_PROBS, PAST_DELTA, survival_probs,
                     hsmm.trans_mat, T, N, D)

    return _backtracking(delta, psi_state, psi_dur, T)
