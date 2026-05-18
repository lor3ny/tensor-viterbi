import copy
import random
import numpy as np
import json

class HSMM:
    def __init__(self, states, emissions=None, trans_mat=None, emission_prob=None,
                 duration_probs_linear=None, start_probs=None, duration_probs=None):
        self.states = states
        self.emissions = emissions
        self.trans_mat = trans_mat
        self.emission_probs = emission_prob
        self.duration_probs_linear = duration_probs_linear
        self.start_probs = start_probs
        self.duration_probs = duration_probs
        self.obs_seq = None

    # ------------------------------------------------------------------
    # Builder setters
    # ------------------------------------------------------------------

    def set_transitions(self, trans_mat: np.ndarray) -> "HSMM":
        """Transition matrix, shape (N, N). Rows must sum to 1."""
        self.trans_mat = np.asarray(trans_mat, dtype=float)
        return self

    def set_emissions(self, emissions, emission_probs: np.ndarray) -> "HSMM":
        """Emission symbols and probability matrix, shape (O, N)."""
        self.emissions = emissions
        self.emission_probs = np.asarray(emission_probs, dtype=float)
        return self

    def set_duration_probs(self, duration_probs: np.ndarray) -> "HSMM":
        """Duration probabilities in linear space, shape (D, N).
        Stored as both duration_probs (later log-converted) and
        duration_probs_linear (kept in linear space for native backends)."""
        arr = np.asarray(duration_probs, dtype=float)
        self.duration_probs = arr
        self.duration_probs_linear = arr
        return self

    def set_start_probs(self, start_probs: np.ndarray) -> "HSMM":
        """Initial state distribution, shape (N,)."""
        self.start_probs = np.asarray(start_probs, dtype=float)
        return self

    def set_observations(self, obs_seq: np.ndarray) -> "HSMM":
        """Observation sequence, shape (T,)."""
        self.obs_seq = np.asarray(obs_seq, dtype=float)
        return self

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    def _missing(self) -> list[str]:
        required = {
            "transitions (set_transitions)":     self.trans_mat,
            "emissions (set_emissions)":          self.emission_probs,
            "duration probs (set_duration_probs)": self.duration_probs,
            "start probs (set_start_probs)":      self.start_probs,
            "observations (set_observations)":    self.obs_seq,
        }
        return [name for name, val in required.items() if val is None]

    def is_complete(self) -> bool:
        return len(self._missing()) == 0

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self) -> np.ndarray:
        """Run Viterbi decoding. Raises if any required field is not set."""
        missing = self._missing()
        if missing:
            raise RuntimeError(
                "HSMM model is incomplete. Missing fields:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )
        from tensor_viterbi.viterbi.tensor import decode_log_tensor_viterbi_cached
        h = copy.copy(self)
        h.to_log_space()
        return decode_log_tensor_viterbi_cached(h)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_obs_sequence(self, obs_seq):
        self.obs_seq = obs_seq

    def to_log_space(self):
        smoothness = 1e-30
        self.trans_mat = np.log(self.trans_mat + smoothness)
        self.emission_probs = np.log(self.emission_probs + smoothness)
        self.start_probs = np.log(self.start_probs + smoothness)
        self.duration_probs = np.log(self.duration_probs + smoothness)

    def print_model(self):
        N = len(self.states)
        O = len(self.emissions) if self.emissions is not None else "?"
        D = self.duration_probs.shape[0] if self.duration_probs is not None else "?"
        T = len(self.obs_seq) if self.obs_seq is not None else "?"

        print("===== HSMM MODEL =====")
        print(f"\nDimensions:")
        print(f"  N (states)    = {N}")
        print(f"  O (emissions) = {O}")
        print(f"  D (max dur)   = {D}")
        print(f"  T (obs len)   = {T}")

        print(f"\nStates ({N}):")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s}")

        if self.start_probs is not None:
            print("\nStart probabilities (pi):")
            for i, s in enumerate(self.states):
                print(f"  {s}: {self.start_probs[i]:.6f}")

        if self.trans_mat is not None:
            print("\nTransition matrix (N x N):")
            for i in range(N):
                row = "  ".join(f"{self.trans_mat[i, j]:8.6f}" for j in range(N))
                print(f"  {row}")

        if self.emission_probs is not None:
            print(f"\nEmission probabilities (O x N):")
            for o in range(len(self.emissions)):
                row = "  ".join(f"{self.emission_probs[o, s]:8.6f}" for s in range(N))
                print(f"  Obs {o}: {row}")

        if self.duration_probs is not None:
            D = self.duration_probs.shape[0]
            print("\nDuration probabilities:")
            for s in range(N):
                row = "  ".join(f"{self.duration_probs[d, s]:.6f}" for d in range(D))
                print(f"  State {self.states[s]}: [ {row} ]")

        print("\n======================")

    @staticmethod
    def load_model(json_path: str = "hsmm_config.json") -> "HSMM":
        with open(json_path, "r") as f:
            cfg = json.load(f)

        n_bins       = int(cfg["n_bins"])
        seed         = int(cfg["seed"])

        np.random.seed(seed)
        random.seed(seed)

        sleep_states = [s["name"] for s in cfg["states"]]
        sleep_emissions = np.arange(n_bins)
        sleep_obs_seq = np.array(cfg["obs_seq"], dtype=float) - 1

        sleep_trans_mat = np.array(cfg["trans_mat"], dtype=float)

        emission_by_state = np.array(
            [s["emission_probs"] for s in cfg["states"]], dtype=float
        )
        sleep_emission_probs = emission_by_state.T

        sleep_start_probs = np.array(cfg["pi"], dtype=float)

        sleep_duration_probs = np.array(
            [s["duration_probs"] for s in cfg["states"]], dtype=float
        )

        hsmm_sleep = HSMM(
            sleep_states,
            sleep_emissions,
            sleep_trans_mat.T,
            sleep_emission_probs,
            sleep_duration_probs.T,
            sleep_start_probs,
            sleep_duration_probs.T,
        )
        hsmm_sleep.set_obs_sequence(sleep_obs_seq)

        return hsmm_sleep
