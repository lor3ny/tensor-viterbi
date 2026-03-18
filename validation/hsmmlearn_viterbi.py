import json
import random
import numpy as np
import time
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel


class MultinomialEmissions(AbstractEmissions):
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
        return MultinomialEmissions(self._emission_matrix.copy())


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


    emissions = MultinomialEmissions(emission_matrix)

    model = HSMMModel(
        emissions,
        duration_matrix,
        tmat,
        startprob=startprob
    )

    return model, obs_seq

def compute_accuracy(true_states, predicted_states):
    true_states = np.array(true_states)
    predicted_states = np.array(predicted_states)
    return np.sum(true_states == predicted_states) / len(true_states)


def validate(computed_states: np.ndarray, json_file: str):
    model, obs_seq = load_sleep_model_hsmmlearn(json_file)

    start_time = time.time()
    decoded_states = model.decode(obs_seq)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Baseline HSMMLearn Viterbi: {execution_time:.4f} seconds")

    acc = compute_accuracy(decoded_states, computed_states)
    print(f"Accuracy: {acc:.2%}") 
