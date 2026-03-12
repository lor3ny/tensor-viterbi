from curses import window
import random
import numpy as np
import matplotlib.pyplot as plt
import time


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
    

    def compute_ap(self,A, P):
        """
        Compute AP = OuterProduct(A[i,j], P[i,d])
        A: NxN matrix
        P: NxD matrix
        AP: NxNxD tensor
        """
        # AP[i,j,d] = A[i,j] * P[i,d]
        return A[:, :, np.newaxis] * P[np.newaxis, :, :]  # (N,N,1) * (N,1,D) -> NxNxD


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
        smoothing = 1e-10
        
        delta = np.full((T, N), 0.0)        
        delta_state = np.zeros((T, N), dtype=int)
        delta_dur = np.zeros((T, N), dtype=int)
        

        #* INITIALIZATION

        AP = np.zeros((N, N, D)) # Precompute AP outside the loop
        
        AP[:, :, :] = self.start_probs[np.newaxis, :, np.newaxis]
     
        AP *= self.emission_probs[int(self.obs_seq[0]), np.newaxis, :, np.newaxis]

        (p_maxs, s_maxs, d_maxs) = self.find_t_maxs(AP) # In questo caso non serve, ma lo calcoliamo per verificare che sia tutto ok
        delta[0, :] = p_maxs 
        delta_state[0, :] = np.array((-1,-1,-1,-1))
        delta_dur[0, :] = np.array((1,1,1,1))
        

        #* INDUCTION

        AP = self.compute_ap(self.trans_mat, self.duration_probs)  # NxNxD — precomputed outside the loop


        PAST_DELTA = np.zeros((N, D))
        EMISSION_PROBS = np.zeros((N, D)) # Placeholder: replace with real emission computation
        DELTA_EMISSION = np.zeros((N, N, D))

        for t in range(1, T):
            # Slice DELTAS window: shape (N, D) assuming DELTAS is shape (T, N, D)

            for d_val in range(1, min(D, t+1)):
                segment_indices = np.array(self.obs_seq[t - d_val : t], dtype=int)
                # # 2. Extract the relevant rows from the emission matrix
                # # This creates a sub-matrix of shape (d, num_states)
                relevant_probs = self.emission_probs[segment_indices, :]   #DxN
                  # # 3. Multiply along the 'duration' axis (axis 0)
                # # This collapses the (d, num_states) matrix into a (num_states,) vector
                EMISSION_PROBS[:, d_val - 1] = np.prod(relevant_probs, axis=0)


            window = delta[max(0, t-D) : t, :]  # shape: (min(t,D), N)
            PAST_DELTA[:, :window.shape[0]] = window[::-1].T 

            #! emission prob computation
            # EMISSION_PROBABILITY = np.ones((N, D))             # 

            # # Method A
            # DELTA_EMISSION = PAST_DELTA[:, np.newaxis, :] * EMISSION_PROBS[np.newaxis, :, :]  # NxNxD
            # RESULT_A = AP #* DELTA_EMISSION  # NxNxD element-wise

            #print(RESULT_A)

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
        smoothing = 1e-10
        
        # Delta: max prob ending at t in state j
        delta = np.full((T, N), -np.inf)
        
        # Backpointers to reconstruct path
        # psi_state[t, j] = previous state i that led to j ending at t
        # psi_dur[t, j] = duration d that state j held ending at t
        psi_state = np.zeros((T, N), dtype=int)
        psi_dur = np.zeros((T, N), dtype=int)


        #* INITIALIZATION  t==0
        #* delta(0,sj) = pi(sj) * b(sj, obs_seq[1])

        #! The gemini proposed version was including also duration, but why? 
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

        EMISSION = np.zeros((N, D)) # Placeholder: replace with real emission computation

        for t in range(1, T):
            for sj in range(N):
                for d in range(1, D + 1):
                    if t - d < 0: 
                        continue # Cannot look back past 0 here
                    
                    #! This productory can be optimized and precomputed
                    # |-|{k = t-d}(b(sj, seq_obs(k)
                    obs_score = 1.0
                    for k in range(d):
                        obs_index = int(self.obs_seq[t-k-1])
                        obs_score *= self.emission_probs[obs_index, sj]

                    EMISSION[sj, d-1] = obs_score
                    
                    # P(d|Sj)
                    dur_score = self.duration_probs[sj, d-1]
                    
                    best_prev_score = -np.inf
                    best_prev_state = -1
                    for si in range(N):

                        # HSMMs handle self-loops via duration, Skip impossibile transitions
                        # if si == sj or self.trans_mat[si, sj] == 0: 
                        #     continue 
                        # 
                        
                        # a(si,sj)
                        trans_score = self.trans_mat[si, sj]      # Delta: max prob ending at t in state jng)

                        # Score = delta(t-d,si) + Transition + Duration + Emissions
                        total_score = trans_score * dur_score * delta[t - d, si] * obs_score

                        # print(trans_score, dur_score, delta[t - d, si], total_score)

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
    

if __name__ == "__main__":


    # States: 0=Awake, 1=Light, 2=Deep, 3=REM
    sleep_states = ["Awake", "Light", "Deep", "REM"]

    # Emissions: Heart Rate (HR) discretized into 13 bins (0-12)
    # 0-4: Very Low/Stable, 5-8: Moderate, 9-12: High/Variable
    sleep_emissions = np.arange(13)

    time_steps = 1000
    max_duration = 100 # Increased to accommodate longer sleep cycles

    sleep_obs_seq = np.zeros(time_steps)
    for i in range(time_steps):
        if i < 30: # Deep Sleep
            sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4]) # Low HR, low variance
        elif i < 70: # REM Sleep
            sleep_obs_seq[i] = random.choice([5, 6, 7, 8, 9]) # High HR, high variance
        else: # Deep Sleep again
            sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4])
    print("Generated Observations (Heart Rate):", sleep_obs_seq)



    # --- 2. TRANSITION MATRIX (A) ---
    # Diagonal must be 0.0. Transitions represent where you go AFTER a duration ends.
    # Structure: [To Awake, To Light, To Deep, To REM]
    sleep_trans_mat = np.array([
        [0.0, 0.9, 0.0, 0.1], # From Awake: Mostly to Light
        [0.1, 0.0, 0.5, 0.4], # From Light: To Deep, REM, or briefly Awake
        [0.0, 1.0, 0.0, 0.0], # From Deep: Almost always back to Light first
        [0.2, 0.8, 0.0, 0.0]  # From REM: To Light or wake up
    ])

    # --- 3. EMISSION PROBABILITIES (B) ---
    # Each column represents a state; rows represent the 13 HR bins.
    # We use a distribution where Deep is low HR, Awake/REM are higher HR.
    sleep_emission_probs = np.array([
        # HR Bin | Awake | Light | Deep  | REM
        [0.001,  0.010,  0.150,  0.001], # Bin 0 (Lowest HR)
        [0.001,  0.030,  0.300,  0.001],
        [0.002,  0.100,  0.350,  0.002],
        [0.002,  0.250,  0.150,  0.002],
        [0.002,  0.300,  0.040,  0.002], # Bin 4
        [0.010,  0.200,  0.005,  0.010],
        [0.030,  0.080,  0.002,  0.030],
        [0.150,  0.020,  0.001,  0.200], # Bin 7 (Moderate)
        [0.300,  0.005,  0.001,  0.400],
        [0.300,  0.002,  0.001,  0.250],
        [0.150,  0.001,  0.000,  0.080],
        [0.040,  0.001,  0.000,  0.020],
        [0.012,  0.001,  0.000,  0.002], # Bin 12 (Highest HR)
    ])

    # --- 4. START & DURATION PROBABILITIES ---

    # Most people start "Awake" (1.0) or "Light Sleep" (0.0)
    sleep_start_probs = np.array([0.9, 0.1, 0.0, 0.0])

    def gaussian_window(length, mean, std):
        x = np.arange(length)
        # Using LaTeX for the Gaussian distribution logic
        # $$G(x) = \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
        g = np.exp(-0.5 * ((x - mean) / std)**2)
        return g / g.sum()

    sleep_duration_probs = np.zeros((4, max_duration))

    # Define distinct durations for each state
    sleep_duration_probs[0, :] = gaussian_window(max_duration, mean=5,  std=2)  # Awake: Short bursts
    sleep_duration_probs[1, :] = gaussian_window(max_duration, mean=30, std=8)  # Light: Moderate
    sleep_duration_probs[2, :] = gaussian_window(max_duration, mean=50, std=10) # Deep: Long
    sleep_duration_probs[3, :] = gaussian_window(max_duration, mean=25, std=5)  # REM: Medium

    print("Transition Matrix:\n", sleep_trans_mat)
    print("Emission Probabilities:\n", sleep_emission_probs)
    print("Duration Probabilities:\n", sleep_duration_probs)

    hsmm_sleep = HSMM(sleep_states, sleep_emissions, sleep_trans_mat, sleep_emission_probs, sleep_start_probs, sleep_duration_probs)
    hsmm_sleep.set_obs_sequence(sleep_obs_seq)

    start_time = time.time()
    predicted_states = hsmm_sleep.run_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time of Vanilla Viterbi: {execution_time:.4f} seconds")
    print("Predicted States:")
    print(predicted_states)

    start_time = time.time()
    predicted_states = hsmm_sleep.run_tensor_viterbi()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time of Tensor Viterbi: {execution_time:.4f} seconds")
    print("Predicted States:")
    print(predicted_states)


