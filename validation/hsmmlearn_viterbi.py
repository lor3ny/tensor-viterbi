import json
import os
import csv
import random
import numpy as np
import time
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel

# ── ANSI colors (disabled automatically when output is not a terminal) ────────
import sys
_TTY = sys.stdout.isatty()
def _c(code): return code if _TTY else ""

R     = _c("\033[0m")
BOLD  = _c("\033[1m")
GREEN = _c("\033[92m")
GRAY  = _c("\033[90m")
WHITE = _c("\033[97m")
YEL   = _c("\033[93m")
# ─────────────────────────────────────────────────────────────────────────────



class RawEmissions(AbstractEmissions):
    """
    Discrete (categorical) emissions.
    emission_matrix[s, k] = P(obs = k | state = s)  — shape (N, n_bins)
    """
    def __init__(self, emission_matrix):
        self._emission_matrix = emission_matrix  # shape (N, n_bins)

    def likelihood(self, obs):
        # returns shape (N, T) as required by hsmmlearn
        return self._emission_matrix[:, obs.astype(int)]

    def copy(self):
        return RawEmissions(self._emission_matrix.copy())



def compute_accuracy(true_states, predicted_states):
    true_states = np.array(true_states)
    predicted_states = np.array(predicted_states)
    return np.sum(true_states == predicted_states) / len(true_states)



def load_sleep_model_hsmmlearn(json_path: str = "hsmm_config.json"):
    with open(json_path, "r") as f:
        cfg = json.load(f)

    seed = int(cfg["seed"])
    np.random.seed(seed)
    random.seed(seed)


    obs_seq = np.array(cfg["obs_seq"], dtype=float) - 1
    obs_seq = obs_seq.astype(int)


    emission_matrix = np.array(
        [s["emission_probs"] for s in cfg["states"]], dtype=float
    )  # shape (N, n_bins)


    duration_matrix = np.array(
        [s["duration_probs"] for s in cfg["states"]], dtype=float
    )  # shape (N, M)


    tmat      = np.array(cfg["trans_mat"], dtype=float)  # shape (N, N)
    startprob = np.array(cfg["pi"],        dtype=float)  # shape (N,)


    emissions = RawEmissions(emission_matrix)

    model = HSMMModel(
        emissions,
        duration_matrix,
        tmat,
        startprob=startprob
    )

    return model, obs_seq


#! HOOK
#! ---------------------
def benchmark_baseline(json_file: str, csv_path="benchmark.csv", iterations=100, n_states=0, timesteps=0, max_duration=0):

    model, obs_seq = load_sleep_model_hsmmlearn(json_file)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        decoded_states = model.decode(obs_seq)
        times.append(time.perf_counter() - start)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
        for i, t in enumerate(times):
            writer.writerow(["HSMMLearn_CPP", n_states, timesteps, max_duration, i, f"{t:.6f}"])

    avg, mn, mx = sum(times) / len(times), min(times), max(times)
    print(f"  {WHITE}HSMMLearn C++{R}")
    print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")
    return

def measure_baseline(json_file: str):
    model, obs_seq = load_sleep_model_hsmmlearn(json_file)
    start_time = time.perf_counter()
    decoded_states = model.decode(obs_seq)
    elapsed = time.perf_counter() - start_time
    print(f"  {GRAY}time{R}  {WHITE}HSMMLearn C++{R}  {BOLD}{GREEN}{elapsed:.4f} s{R}\n")
    return elapsed

def validate(title_str: str, computed_states: np.ndarray, json_file: str, print_states: bool = False):
    model, obs_seq = load_sleep_model_hsmmlearn(json_file)

    decoded_states = model.decode(obs_seq)
    
    if(print_states):
        print(f'HSMMLearn Decoded States: {decoded_states}')
        
    acc = compute_accuracy(decoded_states, computed_states)
    acc_color = GREEN if acc >= 0.95 else YEL
    print(f"  {GRAY}accuracy{R}  {WHITE}{title_str}{R}  {BOLD}{acc_color}{acc:.2%}{R}\n")
