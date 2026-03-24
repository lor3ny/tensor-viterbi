from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from deprecated import deprecated

if TYPE_CHECKING:
    from hsmm import HSMM


def _backtracking(delta: np.ndarray, psi_state: np.ndarray, psi_dur: np.ndarray, T: int) -> np.ndarray:
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


@deprecated(reason="It doesn't work after 370 timesteps because it goes on underflow, use the log-space function.")
def decode_tensor_viterbi(hsmm: HSMM) -> np.ndarray:
    hsmm.duration_probs = hsmm.duration_probs.T
    hsmm.trans_mat = hsmm.trans_mat.T

    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), 0.0)
    delta_state = np.zeros((T, N), dtype=int)
    delta_dur = np.zeros((T, N), dtype=int)

    EMISSION_PROBS = np.ones((D, N))
    DELTA_EMISSION = np.ones((N, N, D))
    AP = np.ones((N, N, D))

    #! PHASE 1 - INITIALIZATION 0<=t<D
    PAST_DELTA = hsmm.duration_probs * hsmm.start_probs[np.newaxis, :]

    obs_indices = hsmm.obs_seq[:D].astype(int)
    emission_rows = hsmm.emission_probs[obs_indices, :]
    cum_emission = np.cumprod(emission_rows, axis=0)
    EMISSION_PROBS = cum_emission

    delta[0:D] = PAST_DELTA * EMISSION_PROBS
    delta_dur[0:D] = np.arange(1, D + 1)[:, np.newaxis] * np.ones((1, N), dtype=int)

    #! PHASE 2 - INDUCTION  t>0
    AP = hsmm.trans_mat[np.newaxis, :, :] * hsmm.duration_probs[:, :, np.newaxis]
    for t in range(1, T):
        segment_indices = hsmm.obs_seq[max(0, t - D + 1) : t + 1].astype(int)
        relevant_probs = hsmm.emission_probs[segment_indices, :]
        cum_emission = np.cumprod(np.flip(relevant_probs, axis=0), axis=0)
        EMISSION_PROBS[: cum_emission.shape[0], :] = cum_emission

        window = delta[max(0, t - D) : t, :]
        PAST_DELTA[: window.shape[0], :] = window[::-1]

        DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * AP
        RESULT_B = EMISSION_PROBS[:, :, np.newaxis] * DELTA_EMISSION

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

    return _backtracking(delta, delta_state, delta_dur, T)


@deprecated(reason="It is slower than the cached version.")
def decode_log_tensor_viterbi_no_cache(hsmm: HSMM) -> np.ndarray:
    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), -np.inf)
    delta_state = np.zeros((T, N), dtype=int)
    delta_dur = np.zeros((T, N), dtype=int)

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

    return _backtracking(delta, delta_state, delta_dur, T)


#? We use the official signature of Numpy for 3D tensors (d,y,x)
#? axis 0 → depth (z) — the first index, selects a 2D "slice"
#? axis 1 → rows — the second index, selects a row within a slice
#? axis 2 → columns — the third index, selects a column within a row
def decode_log_tensor_viterbi_cached(hsmm: HSMM) -> np.ndarray:
    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), -np.inf)
    delta_state = np.zeros((T, N), dtype=int)
    delta_dur = np.zeros((T, N), dtype=int)

    #! PHASE 1 - INITIALIZATION 0<=t<D
    PAST_DELTA = hsmm.duration_probs + hsmm.start_probs[np.newaxis, :]

    obs_indices = hsmm.obs_seq[:D].astype(int)
    emission_rows = hsmm.emission_probs[obs_indices, :]
    cum_emission = np.cumsum(emission_rows, axis=0)
    EMISSION_PROBS = cum_emission

    delta[0:D] = PAST_DELTA + EMISSION_PROBS
    delta_dur[0:D] = np.arange(1, D + 1)[:, np.newaxis] * np.ones((1, N), dtype=int)

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
        delta_state[t, :] = np.where(cond, delta_state[t, :], i_arr)
        delta_dur[t, :] = np.where(cond, delta_dur[t, :], d_arr + 1)

    return _backtracking(delta, delta_state, delta_dur, T)


def decode_vanilla_viterbi(hsmm: HSMM) -> np.ndarray:
    T = len(hsmm.obs_seq)
    N = len(hsmm.states)
    D = hsmm.duration_probs.shape[0]

    delta = np.full((T, N), -np.inf)
    psi_state = np.zeros((T, N), dtype=int)
    psi_dur = np.zeros((T, N), dtype=int)

    #! PHASE 1 - INITIALIZATION 0<=t<D
    for state in range(N):
        for d in range(1, D + 1):
            obs_score = 0.0
            for tau in range(0, d):
                obs_index = int(hsmm.obs_seq[tau])
                obs_score += hsmm.emission_probs[obs_index, state]

            dur_score = hsmm.duration_probs[d - 1, state]
            start_prob = hsmm.start_probs[state]
            score = start_prob + dur_score + obs_score
            if score > delta[d - 1, state]:
                delta[d - 1, state] = score
                psi_dur[d - 1, state] = d
                psi_state[d - 1, state] = state

    #! PHASE 2 - INDUCTION  t>0
    for t in range(1, T):
        for sj in range(N):
            for d in range(1, D + 1):
                if t - d < 0:
                    continue

                obs_score = 0.0
                for tau in range(t - d, t + 1):
                    obs_index = int(hsmm.obs_seq[tau])
                    obs_score += hsmm.emission_probs[obs_index, sj]

                dur_score = hsmm.duration_probs[d - 1, sj]

                best_prev_score = -np.inf
                best_prev_state = -1
                for si in range(N):
                    total_score = hsmm.trans_mat[si, sj] + dur_score + delta[t - d, si] + obs_score

                    if total_score > best_prev_score:
                        best_prev_score = total_score
                        best_prev_state = si

                if best_prev_score > delta[t, sj]:
                    delta[t, sj] = best_prev_score
                    psi_state[t, sj] = best_prev_state
                    psi_dur[t, sj] = d

    return _backtracking(delta, psi_state, psi_dur, T)
