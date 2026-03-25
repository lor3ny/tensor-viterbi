from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .tensor import _backtracking

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM


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
                for tau in range(t - d + 1, t + 1):
                    obs_index = int(hsmm.obs_seq[tau])
                    obs_score += hsmm.emission_probs[obs_index, sj]

                dur_score = hsmm.duration_probs[d - 1, sj]

                best_prev_score = -np.inf
                best_prev_state = -1
                for si in range(N):
                    total_score = hsmm.trans_mat[sj, si] + dur_score + delta[t - d, si] + obs_score

                    if total_score > best_prev_score:
                        best_prev_score = total_score
                        best_prev_state = si

                if best_prev_score > delta[t, sj]:
                    delta[t, sj] = best_prev_score
                    psi_state[t, sj] = best_prev_state
                    psi_dur[t, sj] = d

    return _backtracking(delta, psi_state, psi_dur, T)
