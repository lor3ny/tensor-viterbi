import random
import numpy as np
import json

#from tensor_viterbi import run_log_tensor_viterbi_cached, run_vanilla_viterbi, run_log_tensor_viterbi_no_cache, run_tensor_viterbi

#! ----------------------------------------------------
#! HSMM CLASS
#!
#! ----------------------------------------------------

class HSMM:
    def __init__(self, states, emissions, trans_mat, emission_prob, start_probs, duration_probs):
            self.states = states
            self.emissions = emissions
            self.trans_mat = trans_mat
            self.emission_probs = emission_prob
            self.start_probs = start_probs
            self.duration_probs = duration_probs

    def set_obs_sequence(self, obs_seq):
        self.obs_seq = obs_seq

    def print_model(self):
        N = len(self.states)
        O = len(self.emissions)
        D = self.duration_probs.shape[0]
        T = len(self.obs_seq)

        print("===== HSMM MODEL =====")

        # Dimensioni
        print("\nDimensions:")
        print(f"  N (states)    = {N}")
        print(f"  O (emissions) = {O}")
        print(f"  D (max dur)   = {D}")
        print(f"  T (obs len)   = {T}")

        # Stati
        print(f"\nStates ({N}):")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s}")

        # Emissioni
        print(f"\nEmissions ({O}):")
        for o, e in enumerate(self.emissions):
            print(f"  [{o}] {e}")

        # Start probabilities
        print("\nStart probabilities (pi):")
        for i, s in enumerate(self.states):
            print(f"  {s}: {self.start_probs[i]:.6f}")

        # Transition matrix
        print("\nTransition matrix (N x N):")
        for i in range(N):
            row = "  ".join(f"{self.trans_mat[i, j]:8.6f}" for j in range(N))
            print(f"  {row}")

        # Emission probabilities
        print("\nEmission probabilities (O x N):")
        for o in range(O):
            row = "  ".join(f"{self.emission_probs[o, s]:8.6f}" for s in range(N))
            print(f"  Obs {o}: {row}")

        # Duration probabilities
        print("\nDuration probabilities:")
        for s in range(N):
            row = "  ".join(f"{self.duration_probs[d, s]:.6f}" for d in range(D))
            print(f"  State {self.states[s]}: [ {row} ]")

        print("\n======================")

    @staticmethod
    def load_model(json_path: str = "hsmm_config.json") -> HSMM:

        with open(json_path, "r") as f:
            cfg = json.load(f)
    
        # ── scalars ──────────────────────────────────────────────────────────────
        time_steps   = int(cfg["n_steps"])
        max_duration = int(cfg["M"])
        n_bins       = int(cfg["n_bins"])
        seed         = int(cfg["seed"])
    
        np.random.seed(seed)
        random.seed(seed)
    
        sleep_states = [s["name"] for s in cfg["states"]]   # ["Awake", ...]
        J = len(sleep_states)
    
        sleep_emissions = np.arange(n_bins)                  # shape (13,)
    
        sleep_obs_seq = np.array(cfg["obs_seq"], dtype=float) - 1   # shape (100,)

        sleep_trans_mat = np.array(cfg["trans_mat"], dtype=float)         # shape (4, 4)
    
        emission_by_state = np.array(
            [s["emission_probs"] for s in cfg["states"]], dtype=float
        )                                                            # shape (4, 13)
        sleep_emission_probs = emission_by_state.T                   # shape (13, 4)
    
        sleep_start_probs = np.array(cfg["pi"], dtype=float)        # shape (4,4)
    
        sleep_duration_probs = np.array(
            [s["duration_probs"] for s in cfg["states"]], dtype=float
        )      
        

        smoothness = 1e-30
        hsmm_sleep = HSMM(
            sleep_states, 
            sleep_emissions, 
            np.log(sleep_trans_mat.T + smoothness), 
            np.log(sleep_emission_probs + smoothness), 
            np.log(sleep_start_probs + smoothness), 
            np.log(sleep_duration_probs.T + smoothness)
        )
        hsmm_sleep.set_obs_sequence(sleep_obs_seq)

        return hsmm_sleep
