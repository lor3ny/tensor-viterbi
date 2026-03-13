from curses import window
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import json


# --- 3. HELPER FUNCTIONS ---

def get_log_emission_probs(obs, means, stds):
    """Calculate log P(observation | state) for all t, all states."""
    T = len(obs)
    n = len(means)
    log_probs = np.zeros((T, n))
    for s in range(n):
        # Log PDF of Gaussian
        var = stds[s]**2
        denom = np.sqrt(2 * np.pi * var)
        exponent = -0.5 * ((obs - means[s])**2) / var
        log_probs[:, s] = exponent - np.log(denom)
    return log_probs

# --- 4. HSMM VITERBI ALGORITHM ---


# obs: observations sequence of length T
# trans_mat: N x N transition matrix (0 on diagonal for HSMM)
# duration_probs: N x D matrix of duration probabilities
# means, stds: N-length arrays for emission distributions

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


    # This function is helpful if you don't have the emission probs, but you have only a mean ad std value.
    def gen_log_emission_probs(self, means, stds):
        """Calculate log P(observation | state) for all t, all states."""
        T = len(self.obs_seq)
        n = len(self.states)
        log_probs = np.zeros((T, n))
        for s in range(n):
            # Log PDF of Gaussian
            var = stds[s]**2
            denom = np.sqrt(2 * np.pi * var)
            exponent = -0.5 * ((self.obs_seq - means[s])**2) / var
            log_probs[:, s] = exponent - np.log(denom)
        return log_probs


    def find_t_maxs(self, Sjid):
        """
        t_MAXs: foreach S[j]-plane (i.e., for each j), find MAX over (i, d)
        RESULT shape: (N, N, D)
        
        Returns:
            max_vals:   (N,) — max probability per j-plane
            max_states: (N,) — i coordinate (state) of the max per j-plane
            max_durs:   (N,) — d coordinate (duration) of the max per j-plane
        """
        N = Sjid.shape[1]
        
        # Reshape each j-plane (N, D) into a flat array, find argmax, then unravel
        max_vals   = np.zeros(N)
        max_states = np.zeros(N, dtype=int)
        max_durs   = np.zeros(N, dtype=int)

        for j in range(N):
            plane = Sjid[:, j, :]          # shape (N, D) — the j-th plane
            flat_idx = np.argmax(plane)      # argmax over flattened (N*D)
            i, d = np.unravel_index(flat_idx, plane.shape)  # recover (i, d) coords
            max_vals[j]   = plane[i, d]
            max_states[j] = i               # x coordinate = source state
            max_durs[j]   = d               # y coordinate = duration

        return max_vals, max_states, max_durs


    def backtracking_termination(self, delta, psi_state, psi_dur, T):
        #! THIS SECTION CAN BE PORTED ON CPU
        #* TERMINATION
        path = np.zeros(T, dtype=int)
        
        # Find best ending state at T-1
        t = T - 1
        best_last_state = np.argmax(delta[t])
        curr_state = best_last_state
        
        while t >= 0:
            d = psi_dur[t, curr_state]
            prev_s = psi_state[t, curr_state]
            
            # Fill the segment
            start_t = t - d + 1
            path[start_t : t+1] = curr_state
            
            # Move back
            t = t - d
            curr_state = prev_s
        return path

    def run_tensor_viterbi(self):
        T = len(self.obs_seq)  # time steps
        N = len(self.states) # states count
        D = self.duration_probs.shape[1]
        
        delta = np.full((T, N), 0.0)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)

        PAST_DELTA = np.zeros((N, D))
        EMISSION_PROBS = np.zeros((N, D))
        DELTA_EMISSION = np.zeros((N, N, D))
        AP = np.zeros((N, N, D)) # Precompute AP outside the loop
        

        #* INITIALIZATION
        AP[:, :, :] = self.start_probs[np.newaxis, :, np.newaxis]
     
        AP *= self.emission_probs[int(self.obs_seq[0]), np.newaxis, :, np.newaxis]

        (p_maxs, s_maxs, d_maxs) = self.find_t_maxs(AP)  #! In questo caso non serve, ma lo calcoliamo per verificare che sia tutto ok
        delta[0, :] = p_maxs 
        delta_state[0, :] = np.array((-1,-1,-1,-1))
        delta_dur[0, :] = np.array((1,1,1,1))
        
        #* INDUCTION
        """
        Compute AP = OuterProduct(A[i,j], P[i,d])
        A: NxN matrix
        P: NxD matrix
        AP: NxNxD tensor
        """
        AP = self.trans_mat[:, :, np.newaxis] * self.duration_probs[np.newaxis, :, :]  # (N,N,1) * (N,1,D) -> NxNxD
 
        for t in range(1, T):

            # Slice DELTAS window: shape (N, D) assuming DELTAS is shape (T, N, D)
            for d_val in range(1, min(D, t+1)):
                segment_indices = np.array(self.obs_seq[t - d_val : t], dtype=int)
                relevant_probs = self.emission_probs[segment_indices, :]   # DxN
                EMISSION_PROBS[:, d_val - 1] = np.prod(relevant_probs, axis=0)


            window = delta[max(0, t-D) : t, :]  # shape: (min(t,D), N)
            PAST_DELTA[:, :window.shape[0]] = window[::-1].T 

            #! emission prob computation
            # EMISSION_PROBABILITY = np.ones((N, D))             # 

            # # Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :]  # NxNxD
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            # # Method B
            # Step 1: Y_BroadcastProduct — PAST_DELTA (N,D) broadcast over j-axis of AP (N,N,D)
            DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * AP  # (N,1,D) * (N,N,D) -> NxNxD
            # Step 2: X_BroadcastProduct — EMISSION_PROBABILITY (N,D) broadcast over i-axis
            RESULT_B = EMISSION_PROBS[np.newaxis, :, :] * DELTA_EMISSION  # (1,N,D) * (N,N,D) -> NxNxD

            (p_maxs, s_maxs, d_maxs) = self.find_t_maxs(RESULT_B)   
            delta[t, :] = p_maxs 
            delta_state[t, :] = s_maxs
            delta_dur[t, :] = d_maxs+1

        path = self.backtracking_termination(delta, delta_state, delta_dur, T)
        
        return path


    # We use np.log() + smoothing to transform multiplications in additions
    def run_viterbi(self):

        T = len(self.obs_seq)  # time steps
        N = len(self.states) # states count
        D = self.duration_probs.shape[1] - 1   # duration probabilities count
        
        # Delta: max prob ending at t in state j
        delta = np.full((T, N), -np.inf)
        
        # Backpointers to reconstruct path
        # psi_state[t, j] = previous state i that led to j ending at t
        # psi_dur[t, j] = duration d that state j held ending at t
        psi_state = np.zeros((T, N), dtype=int)
        psi_dur = np.zeros((T, N), dtype=int)


        #* INITIALIZATION  t==0
        #* delta(0,sj) = pi(sj) * b(sj, obs_seq[1])
        for state in range(N):
            obs_index = int(self.obs_seq[0])
            obs_prob = self.emission_probs[obs_index, state]
            start_prob = self.start_probs[state]
            
            score = start_prob * obs_prob
            if score > delta[0, state]:
                delta[0, state] = score
                psi_dur[0, state] = 1
                psi_state[0, state] = -1 # Indicates start of sequence

        #* INDUCTION  1<=t<=T
        #* delta(t, sj) = max{d} ( max{si} ( delta(t-d,si) * a(si,sj) ) * P(d|sj) * |-|{k = t-d}(b(sj, seq_obs(k)))  
        for t in range(1, T):
            for sj in range(N):
                for d in range(1, D + 1):
                    if t - d < 0: 
                        continue # Cannot look back past 0 here
                    
                    # |-|{k = t-d}(b(sj, seq_obs(k)
                    obs_score = 1.0
                    for k in range(d):
                        obs_index = int(self.obs_seq[t-k-1])
                        obs_score *= self.emission_probs[obs_index, sj]
                    
                    # P(d|Sj)
                    dur_score = self.duration_probs[sj, d-1]
                    
                    best_prev_score = -np.inf
                    best_prev_state = -1
                    for si in range(N):
                        # HSMMs handle self-loops via duration, Skip impossibile transitions. 
                        #! But with product is necessary, maybe can be inserted but doesn't change much in terms of performance
                        # if si == sj or self.trans_mat[si, sj] == 0: 
                        #     continue

                        # Score = delta(t-d,si) + a(si,sj)-Transition + Duration + Emissions
                        total_score = delta[t - d, si] * self.trans_mat[si, sj] * dur_score * obs_score 

                        if total_score > best_prev_score:
                            best_prev_score = total_score
                            best_prev_state = si
                    
                    # Update Delta if this duration d is better than others for ending at t
                    if best_prev_score > delta[t, sj]:
                        delta[t, sj] = best_prev_score
                        psi_state[t, sj] = best_prev_state
                        psi_dur[t, sj] = d             

        path = self.backtracking_termination(delta, psi_state, psi_dur, T)
            
        return path
    


def load_sleep_model(json_path: str = "hsmm_config.json") -> HSMM:

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
 
    sleep_start_probs = np.array(cfg["pi"], dtype=float)        # shape (4,)
 
    sleep_duration_probs = np.array(
        [s["duration_probs"] for s in cfg["states"]], dtype=float
    )                                                            # shape (4, M)
 
    hsmm_sleep = HSMM(
        sleep_states, 
        sleep_emissions, 
        sleep_trans_mat, 
        sleep_emission_probs, 
        sleep_start_probs, 
        sleep_duration_probs
    )
    hsmm_sleep.set_obs_sequence(sleep_obs_seq)

    return hsmm_sleep

def compute_accuracy(true_states, predicted_states):
    true_states = np.array(true_states)
    predicted_states = np.array(predicted_states)
    return np.sum(true_states == predicted_states) / len(true_states)


if __name__ == "__main__":


    # # States: 0=Awake, 1=Light, 2=Deep, 3=REM
    # sleep_states = ["Awake", "Light", "Deep", "REM"]
    # # Emissions: Heart Rate (HR) discretized into 13 bins (0-12)
    # # 0-4: Very Low/Stable, 5-8: Moderate, 9-12: High/Variable
    # sleep_emissions = np.arange(13)

    # time_steps = 38
    # # --- CONFIGURATION ---
    # max_duration = 30  # HB extended to 30

    # # --- 1. TRANSITION MATRIX (A) ---
    # sleep_trans_mat = np.array([
    #     [0.0, 0.9, 0.0, 0.1],  # From Awake: Mostly to Light
    #     [0.1, 0.0, 0.5, 0.4],  # From Light: To Deep, REM, or briefly Awake
    #     [0.0, 1.0, 0.0, 0.0],  # From Deep: Almost always back to Light first
    #     [0.2, 0.8, 0.0, 0.0]   # From REM: To Light or wake up
    # ])

    # sleep_stat_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]

    # sleep_obs_seq = [18, 17, 14, 14, 12, 20, 17, 18, 10, 13, 13, 13, 14, 15, 15, 14, 15, 13, 14, 14, 5, 7, 7, 3, 7, 5, 3, 11, 12, 9, 6, 4, 8, 2, 0, 5, 1, 3, 2, 3, 3, 1, 7, 4, 6, 5, 4, 5, 3, 6, 6, 6, 9, 6, 6, 7, 4, 9, 4, 17, 8, 4, 3, 4, 4, 0, 2, 0, 5, 3, 6, 6, 8, 8, 10, 7, 4, 8, 8, 2, 3, 2, 0, 0, 0, 3, 2, 2, 5, 1, 2, 4, 1, 0, 1, 1, 6, 9, 9, 5]

    # # --- 2. EMISSION PROBABILITIES (B) ---
    # # Each column represents a state; rows represent the 30 HR bins.
    # # Deep sleep = low HR, Awake/REM = higher HR.
    # sleep_emission_probs = np.array([
    #     # Awake  | Light  | Deep   | REM
    #     [0.0001,  0.005,   0.120,   0.0001],  # Bin 0  (Lowest HR)
    #     [0.0001,  0.010,   0.180,   0.0001],  # Bin 1
    #     [0.0005,  0.020,   0.220,   0.0005],  # Bin 2
    #     [0.001,   0.040,   0.200,   0.001 ],  # Bin 3
    #     [0.001,   0.080,   0.130,   0.001 ],  # Bin 4
    #     [0.001,   0.130,   0.080,   0.001 ],  # Bin 5
    #     [0.002,   0.180,   0.040,   0.002 ],  # Bin 6
    #     [0.002,   0.200,   0.015,   0.002 ],  # Bin 7
    #     [0.003,   0.160,   0.007,   0.003 ],  # Bin 8
    #     [0.005,   0.100,   0.003,   0.005 ],  # Bin 9
    #     [0.010,   0.060,   0.002,   0.010 ],  # Bin 10
    #     [0.020,   0.030,   0.001,   0.025 ],  # Bin 11
    #     [0.040,   0.015,   0.001,   0.060 ],  # Bin 12
    #     [0.080,   0.008,   0.001,   0.130 ],  # Bin 13
    #     [0.120,   0.004,   0.001,   0.200 ],  # Bin 14  (Mid-range)
    #     [0.160,   0.003,   0.000,   0.250 ],  # Bin 15
    #     [0.180,   0.002,   0.000,   0.180 ],  # Bin 16
    #     [0.160,   0.002,   0.000,   0.080 ],  # Bin 17
    #     [0.120,   0.001,   0.000,   0.040 ],  # Bin 18
    #     [0.080,   0.001,   0.000,   0.015 ],  # Bin 19
    #     [0.060,   0.001,   0.000,   0.008 ],  # Bin 20
    #     [0.040,   0.001,   0.000,   0.004 ],  # Bin 21
    #     [0.030,   0.001,   0.000,   0.002 ],  # Bin 22
    #     [0.020,   0.001,   0.000,   0.001 ],  # Bin 23
    #     [0.015,   0.001,   0.000,   0.001 ],  # Bin 24
    #     [0.010,   0.001,   0.000,   0.001 ],  # Bin 25
    #     [0.006,   0.001,   0.000,   0.001 ],  # Bin 26
    #     [0.004,   0.001,   0.000,   0.001 ],  # Bin 27
    #     [0.003,   0.001,   0.000,   0.001 ],  # Bin 28
    #     [0.002,   0.001,   0.000,   0.001 ],  # Bin 29 (Highest HR)
    # ])

    # # Normalize each column so probabilities sum to 1
    # sleep_emission_probs = sleep_emission_probs / sleep_emission_probs.sum(axis=0, keepdims=True)

    # # --- 3. START & DURATION PROBABILITIES ---
    # sleep_start_probs = np.array([0.9, 0.1, 0.0, 0.0])

    # def gaussian_window(length, mean, std):
    #     x = np.arange(length)
    #     # $$G(x) = \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
    #     g = np.exp(-0.5 * ((x - mean) / std) ** 2)
    #     return g / g.sum()

    # sleep_duration_probs = np.zeros((4, max_duration))

    # # Durations scaled for max_duration=30
    # sleep_duration_probs[0, :] = gaussian_window(max_duration, mean=5,  std=2)  # Awake: Short bursts
    # sleep_duration_probs[1, :] = gaussian_window(max_duration, mean=10, std=3)  # Light: Moderate
    # sleep_duration_probs[2, :] = gaussian_window(max_duration, mean=15, std=3)  # Deep:  Long
    # sleep_duration_probs[3, :] = gaussian_window(max_duration, mean=8,  std=2)  # REM:dow(max_duration, mean=3, std=1)  # REM: Medium

    # print("Transition Matrix:\n", sleep_trans_mat)
    # print("Emission Probabilities:\n", sleep_emission_probs)
    # print("Duration Probabilities:\n", sleep_duration_probs)

    # hsmm_sleep = HSMM(sleep_states, sleep_emissions, sleep_trans_mat, sleep_emission_probs, sleep_start_probs, sleep_duration_probs)
    # hsmm_sleep.set_obs_sequence(sleep_obs_seq)

    hsmm_sleep = load_sleep_model("sleep_data.json")

    start_time = time.time()
    predicted_states = hsmm_sleep.run_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Predicted States:")
    print(predicted_states)
    print(f"Execution time of Vanilla Viterbi: {execution_time:.4f} seconds")

    start_time = time.time()
    predicted_states = hsmm_sleep.run_tensor_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time

    print("Predicted States:")
    print(predicted_states)
    print(f"Execution time of Tensor Viterbi: {execution_time:.4f} seconds")


    #acc = compute_accuracy(sleep_stat_seq, predicted_states)
    #print(f"Accuracy: {acc:.2%}") 
