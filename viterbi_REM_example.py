import random
import numpy as np
import matplotlib.pyplot as plt

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
    

    def run_tensor_viterbi(self):
        T = len(self.obs_seq)  # time steps
        N = len(self.states) # states count
        D = self.duration_probs.shape[1] - 1   # duration probabilities count
        smoothing = 1e-10
        
        delta = np.full((T, N), -np.inf)
        
        psi_state = np.zeros((T, N), dtype=int)
        psi_dur = np.zeros((T, N), dtype=int)
        

        #* INITIALIZATION
        for state in range(N):
            obs_prob = self.emission_probs[state][self.obs_seq[0]]
            start_prob = self.start_probs[state]
            
            score = start_prob + obs_prob
            if score > delta[0, state]:
                delta[0, state] = score
                psi_dur[0, state] = 1
                psi_state[0, state] = -1 # Indicates start of sequence

        #* INDUCTION


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
            obs_prob = self.emission_probs[state][self.obs_seq[0]]
            start_prob = self.start_probs[state]
            
            score = start_prob + obs_prob
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
                    
                    #! This productory can be optimized and precomputed
                    # |-|{k = t-d}(b(sj, seq_obs(k)
                    obs_score = 0
                    for k in range(t-d):
                        obs_score += np.log(self.emission_probs[sj][self.obs_seq[k]])
                    
                    # P(d|Sj)
                    dur_score = np.log(self.duration_probs[sj, d] + smoothing)
                    
                    best_prev_score = -np.inf
                    best_prev_state = -1
                    for si in range(N):

                        # HSMMs handle self-loops via duration, Skip impossibile transitions
                        if si == sj or self.trans_mat[si, sj] == 0: 
                            continue 
                        
                        # a(si,sj)
                        trans_score = np.log(self.trans_mat[si, sj] + smoothing)        # Delta: max prob ending at t in state jng)

                        # Score = delta(t-d,si) + Transition + Duration + Emissions
                        total_score = delta[t - d, si] + trans_score + dur_score + obs_score
                        
                        if total_score > best_prev_score:
                            best_prev_score = total_score
                            best_prev_state = si
                    
                    # Update Delta if this duration d is better than others for ending at t
                    if best_prev_score > delta[t, sj]:
                        delta[t, sj] = best_prev_score
                        psi_state[t, sj] = best_prev_state
                        psi_dur[t, sj] = d             



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
    

if __name__ == "__main__":

    # --- 1. CONFIGURATION & DATA GENERATION ---

    # States: 0 = REM (High HR, var), 1 = Deep Sleep (Low HR, stable)
    rem_states = ["REM", "Deep"]
    # Emissions: Heart Rate (HR) in bpm, discretized for simplicity
    rem_emissions = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    time_steps = 100 # Time steps (e.g., minutes of sleep)
    max_duration = 50 # Max duration for any state

    rem_obs_seq = np.zeros(time_steps)
    for i in range(time_steps):
        if i < 30: # Deep Sleep
            rem_obs_seq[i] = random.choice([40, 45, 50, 55, 60]) # Low HR, low variance
        elif i < 70: # REM Sleep
            rem_obs_seq[i] = random.choice([80, 85, 90, 95, 100]) # High HR, high variance
        else: # Deep Sleep again
            rem_obs_seq[i] = random.choice([40, 45, 50, 55, 60])
    print("Generated Observations (Heart Rate):", rem_obs_seq)


    # --- 2. MODEL PARAMETERS ---

    # Transition Matrix (must have 0 on diagonal for HSMM)viterbi
    # We force a switch: If done with REM, go Deep. If done with Deep, go REM.
    rem_trans_mat = np.array([
        [0.0, 1.0], 
        [1.0, 0.0]
    ])

    rem_emission_probs = [
        # REM state
        {
            40: 0.001,
            45: 0.001,
            50: 0.002,
            55: 0.002,
            60: 0.002,
            65: 0.01,
            70: 0.03,
            75: 0.25,
            80: 0.4,
            85: 0.25,
            90: 0.03,
            95: 0.01,
            100: 0.002
        },
        # Deep state
        {
            40: 0.01,
            45: 0.03,
            50: 0.25,
            55: 0.4,
            60: 0.25,
            65: 0.03,
            70: 0.01,
            75: 0.002,
            80: 0.002,
            85: 0.002,
            90: 0.002,
            95: 0.001,
            100: 0.001
        }
    ]

    rem_start_probs = np.array([0.5, 0.5]) # Equal chance to start in either state

    rem_duration_probs = np.array([
        np.zeros(max_duration),
        np.zeros(max_duration)
    ])

    def gaussian_window(length, mean, std):
        x = np.arange(length)
        g = np.exp(-0.5 * ((x - mean) / std)**2)
        return g / g.sum()

    rem_duration_probs[0, :] = gaussian_window(max_duration, mean=20, std=5) # REM duration
    rem_duration_probs[1, :] = gaussian_window(max_duration, mean=40, std=5) # Deep duration


    print("Transition Matrix:\n", rem_trans_mat)
    print("Emission Probabilities:\n", rem_emission_probs)
    print("Duration Probabilities:\n", rem_duration_probs)

    hsmm_sleep = HSMM(rem_states, rem_emissions, rem_trans_mat, rem_emission_probs, rem_start_probs, rem_duration_probs)
    hsmm_sleep.set_obs_sequence(rem_obs_seq)
    predicted_states = hsmm_sleep.run_viterbi()

    print("Predicted States:")
    print(predicted_states)


    # --- 5. RUN AND COMPARE ---
    
    # Visualization in text format
    # print(f"{'Time':<5} | {'Obs (HR)':<10} | {'True':<6} | {'HSMM':<6}")
    # print("-" * 40)
    # for t in range(45, 56): # Inspect the noise spike area
    #     obs_str = f"{rem_obs_seq[t]:.1f}"
    #     is_spike = "<-- SPIKE" if t == 50 else ""
    #     print(f"{t:<5} | {obs_str:<10} | {rem_states[ground_true_states[t]]:<6} | {rem_states[predicted_states[t]]:<6} {is_spike}")

    # print("\nAccuracy:", np.mean(predicted_states == ground_true_states))
