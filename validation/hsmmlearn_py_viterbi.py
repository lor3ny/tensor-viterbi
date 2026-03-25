from __future__ import annotations

import json
import random
import time
import numpy as np
import csv
import os

from validation.hsmmlearn_viterbi import compute_accuracy

_NEG_INF = -np.inf



def CalcStoreD(log_d: np.ndarray, tau: int) -> np.ndarray:

    J, M_plus1 = log_d.shape
    M = M_plus1 - 1

    # Work in probability space for the cumulative sum, then log.
    d = np.exp(log_d)          # shape (J, M+1)

    # D[j, u] = sum_{v=u}^{M} d[j, v]   (suffix sum)
    # We need indices up to tau, so pad if necessary.
    max_u = tau + 1
    d_padded = np.zeros((J, max_u + 1))
    copy_cols = min(M + 1, max_u + 1)
    d_padded[:, :copy_cols] = d[:, :copy_cols]

    # Suffix sum: D[:, u] = sum_{v=u}^{max_u} d_padded[:, v]
    D = np.zeros((J, max_u + 1))
    D[:, max_u] = d_padded[:, max_u]
    for u in range(max_u - 1, 0, -1):
        D[:, u] = D[:, u + 1] + d_padded[:, u]

    # Convert to log, guarding against zeros.
    with np.errstate(divide="ignore"):
        log_D = np.where(D > 0, np.log(D), _NEG_INF)

    return log_D   # shape (J, tau+1)


# ---------------------------------------------------------------------------
# Core Viterbi
# ---------------------------------------------------------------------------
def ViterbiImpl(
    tau: int,
    J: int,
    M: int,
    log_d: np.ndarray,
    log_p: np.ndarray,
    log_pi: np.ndarray,
    log_pdf: np.ndarray,
) -> np.ndarray:

    # ---- pre-compute survival durations D --------------------------------
    log_D = CalcStoreD(log_d, tau)   # shape (J, tau+1)

    # ---- allocate tables -------------------------------------------------
    NEG_INF = _NEG_INF
    alpha = np.full((J, tau), NEG_INF, dtype=float)
    maxU  = np.zeros((J, tau), dtype=int)
    maxI  = np.full((J, tau), -1, dtype=int)

    # ==================================================================
    # PASS 1 — regular time steps  t = 0 … tau-2
    # ==================================================================
    for t in range(tau - 1):           # t = 0 … tau-2
        for j in range(J):
            observ = 0.0               # accumulates sum of log_pdf[j, t-u]
            first_alpha = True

            for u in range(1, min(t, M) + 1):   # u = 1 … min(t, M)
                # Find best predecessor state i (i != j)
                x = NEG_INF
                k = -1
                for i in range(J):
                    if i == j:
                        continue
                    val = log_p[i][j] + alpha[i][t - u]
                    if val > x or k == -1:
                        x = val
                        k = i

                candidate = observ + log_d[j][u] + x
                if first_alpha or candidate > alpha[j][t]:
                    alpha[j][t] = candidate
                    maxU[j][t]  = u
                    maxI[j][t]  = k
                    first_alpha = False

                observ += log_pdf[j][t - u]

            # Initialisation term: segment starts at time 0
            if t + 1 <= M:
                ld_init = log_d[j][t + 1]
                init_val = observ + ld_init + log_pi[j]
                if first_alpha or init_val > alpha[j][t]:
                    alpha[j][t] = init_val
                    maxU[j][t]  = -1
                    maxI[j][t]  = -1

            # Add emission of current observation
            alpha[j][t] += log_pdf[j][t]

    # ==================================================================
    # PASS 2 — final time step  t = tau-1  (right-censored, uses D)
    # ==================================================================
    T = tau - 1
    for j in range(J):
        observ = 0.0
        first_alpha = True

        for u in range(1, tau):       # u = 1 … tau-1
            # Find best predecessor state i (i != j)
            x = NEG_INF
            k = -1
            for i in range(J):
                if i == j:
                    continue
                val = log_p[i][j] + alpha[i][T - u]
                if val > x or k == -1:
                    x = val
                    k = i

            candidate = observ + log_D[j][u] + x
            if first_alpha or candidate > alpha[j][T]:
                alpha[j][T] = candidate
                maxU[j][T]  = u
                maxI[j][T]  = k
                first_alpha = False

            observ += log_pdf[j][T - u]

        # Initialisation term for the final step (segment starts at time 0)
        # Note: C++ uses D[j][tau] (index tau), matching log_D[:, tau].
        init_val = observ + log_D[j][tau] + log_pi[j]
        if first_alpha or init_val > alpha[j][T]:
            alpha[j][T] = init_val
            maxU[j][T]  = -1
            maxI[j][T]  = -1

        alpha[j][T] += log_pdf[j][T]

    # ==================================================================
    # Find best final state
    # ==================================================================
    best_val = NEG_INF
    k = 0
    for j in range(J):
        if alpha[j][T] > best_val:
            best_val = alpha[j][T]
            k = j

    # ==================================================================
    # Backtracking
    # ==================================================================
    hidden_states = np.zeros(tau, dtype=int)

    t = T
    while maxI[k][t] >= 0:
        seg_end   = t
        seg_start = t - maxU[k][t] + 1
        hidden_states[seg_start : seg_end + 1] = k

        k_prev = maxI[k][t]
        t     -= maxU[k][t]
        k      = k_prev

    # Fill the initial segment (goes all the way back to time 0)
    hidden_states[0 : t + 1] = k

    return hidden_states



def load_data(json_path: str = "hsmm_config.json") -> np.ndarray:

    with open(json_path, "r") as f:
        cfg = json.load(f)

    # ── scalars ────────────────────────────────────────────────────────
    tau   = int(cfg["n_steps"])
    M     = int(cfg["M"])
    seed  = int(cfg["seed"])

    np.random.seed(seed)
    random.seed(seed)

    states = cfg["states"]
    J      = len(states)

    obs_seq = np.array(cfg["obs_seq"], dtype=int) - 1   # shape (tau,)

    trans_mat = np.array(cfg["trans_mat"], dtype=float)  # shape (J, J)
    # C++ convention: p[i][j] accessed as p[from][to], same as trans_mat
    with np.errstate(divide="ignore"):
        log_p = np.where(trans_mat > 0, np.log(trans_mat), _NEG_INF)

    pi = np.array(cfg["pi"], dtype=float)                # shape (J,)
    with np.errstate(divide="ignore"):
        log_pi = np.where(pi > 0, np.log(pi), _NEG_INF)


    # Pad a zero column at index 0 so that log_d[j][u] maps directly.
    dur_probs = np.array(
        [s["duration_probs"] for s in states], dtype=float
    )                                                    # shape (J, M)
    log_d = np.full((J, M + 1), _NEG_INF)
    with np.errstate(divide="ignore"):
        log_d[:, 1:] = np.where(dur_probs > 0, np.log(dur_probs), _NEG_INF)


    emission_by_state = np.array(
        [s["emission_probs"] for s in states], dtype=float
    )                                                    # shape (J, n_bins)
    # log_pdf[j, t] = log P(obs_seq[t] | state j)
    with np.errstate(divide="ignore"):
        log_emission = np.where(
            emission_by_state > 0, np.log(emission_by_state), _NEG_INF
        )                                                # shape (J, n_bins)
    log_pdf = log_emission[:, obs_seq]                   # shape (J, tau)


    return tau, J, M, log_d, log_p, log_pi, log_pdf


#! HOOK
#! ---------------------
def validate_py(title_str: str, computed_states: np.ndarray, json_file: str):

    tau, J, M, log_d, log_p, log_pi, log_pdf = load_data(json_file)

    decoded_states = ViterbiImpl(tau, J, M, log_d, log_p, log_pi, log_pdf)

    acc = compute_accuracy(decoded_states, computed_states)
    print(f"{title_str} Accuracy - {acc:.2%}") 


def benchmark_baseline_py(json_file: str, csv_path="benchmark.csv", iterations=100,):

    tau, J, M, log_d, log_p, log_pi, log_pdf = load_data(json_file)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        decoded_states = ViterbiImpl(tau, J, M, log_d, log_p, log_pi, log_pdf)
        times.append(time.perf_counter() - start)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["function", "iteration", "elapsed_s"])
        for i, t in enumerate(times):
            writer.writerow(["HSMMLearn_Python", i, f"{t:.6f}"])

    print(f"HSMMLearn C++: avg={sum(times)/len(times):.4f}s  min={min(times):.4f}s  max={max(times):.4f}s")
    return

def measure_baseline_py(json_file: str):
    tau, J, M, log_d, log_p, log_pi, log_pdf = load_data(json_file)
    start_time = time.time()
    decoded_states = ViterbiImpl(tau, J, M, log_d, log_p, log_pi, log_pdf)
    elapsed = time.perf_counter() - start_time
    print(f"Execution time of HSMMLearn Python: {elapsed:.4f} seconds")
    return
