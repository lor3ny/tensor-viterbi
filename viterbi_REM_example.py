import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & DATA GENERATION ---

# States: 0 = REM (High HR, var), 1 = Deep Sleep (Low HR, stable)
states = ["REM", "Deep"]
n_states = len(states)

# True parameters for generation
# REM: Mean HR 80, Std 5
# Deep: Mean HR 60, Std 3
means = np.array([80, 60])
stds = np.array([5, 3])

# Generate synthetic data
np.random.seed(42)
T = 100 # Total time steps (minutes)

# Create a "Ground Truth" sequence with long durations
# 0-30: REM, 31-70: Deep, 71-100: REM
true_states = np.zeros(T, dtype=int)
true_states[30:70] = 1 
true_states[70:] = 0

# Generate noisy observations (Heart Rate)
observations = np.zeros(T)
for t in range(T):
    state = true_states[t]
    observations[t] = np.random.normal(means[state], stds[state])

# Add a "spike" of noise in Deep sleep that would confuse a normal HMM
# At t=50 (Deep sleep), HR spikes to 90 (looks like REM)
observations[50] = 90 

# --- 2. MODEL PARAMETERS ---

# Transition Matrix (must have 0 on diagonal for HSMM)
# We force a switch: If done with REM, go Deep. If done with Deep, go REM.
# A[i, j] = P(j | i)
trans_mat = np.array([
    [0.0, 1.0], 
    [1.0, 0.0]
])

# Duration Distributions (Non-parametric / Discrete)
# We define max duration D
D = 50 
duration_probs = np.zeros((n_states, D + 1))

# Define simple duration profiles
# REM: prefers short-ish durations (centered around 20)
# Deep: prefers long durations (centered around 40)
# We verify these sum to 1.0 later in a real app, 
# but for this demo we fill them with a simple Gaussian-like window.

def gaussian_window(length, mean, std):
    x = np.arange(length)
    g = np.exp(-0.5 * ((x - mean) / std)**2)
    return g / g.sum()

duration_probs[0, :] = gaussian_window(D + 1, mean=20, std=5) # REM duration
duration_probs[1, :] = gaussian_window(D + 1, mean=40, std=5) # Deep duration

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

def hsmm_viterbi(obs, trans_mat, duration_probs, means, stds):
    T = len(obs)
    N = len(means)
    D = duration_probs.shape[1] - 1
    
    # Precompute emissions
    log_B = get_log_emission_probs(obs, means, stds)
    
    # Precompute CUMULATIVE emissions for O(1) segment scoring
    # pad with 0 at the top for easy indexing
    log_B_cum = np.vstack([np.zeros(N), np.cumsum(log_B, axis=0)])
    
    # Delta: max prob ending at t in state j
    delta = np.full((T, N), -np.inf)
    
    # Backpointers to reconstruct path
    # psi_state[t, j] = previous state i that led to j ending at t
    # psi_dur[t, j] = duration d that state j held ending at t
    psi_state = np.zeros((T, N), dtype=int)
    psi_dur = np.zeros((T, N), dtype=int)
    
    # Initialization (t=0 to D-1 handling is tricky, simplified here)
    # We assume the first segment starts at t=0.
    for d in range(1, min(D, T) + 1):
        # cost of starting in state j with duration d
        # We assume uniform start probability for states (0.5 each)
        start_prob = np.log(0.5) 
        
        for j in range(N):
            dur_prob = np.log(duration_probs[j, d] + 1e-9)
            # Sum of emissions from t=0 to t=d-1
            obs_prob = log_B_cum[d, j] - log_B_cum[0, j]
            
            score = start_prob + dur_prob + obs_prob
            if score > delta[d-1, j]:
                delta[d-1, j] = score
                psi_dur[d-1, j] = d
                psi_state[d-1, j] = -1 # Indicates start of sequence

    # Recursion
    # t is the END time of the current segment
    for t in range(T):
        for j in range(N): # Current state
            # Try all possible durations d for state j
            # segment would be from (t - d + 1) to t
            for d in range(1, D + 1):
                if t - d < 0: continue # Cannot look back past 0 here
                
                prev_t = t - d # Time when previous state ended
                
                # Emission score for this segment (O(1) look up)
                obs_score = log_B_cum[t+1, j] - log_B_cum[t-d+1, j]
                
                # Duration prob
                dur_score = np.log(duration_probs[j, d] + 1e-9)
                
                # Transition from any previous state i to j
                best_prev_score = -np.inf
                best_prev_state = -1
                
                for i in range(N):
                    if i == j: continue # HSMMs handle self-loops via duration
                    if trans_mat[i, j] == 0: continue
                    
                    # Score = Previous Best + Transition + Duration + Emissions
                    trans_score = np.log(trans_mat[i, j] + 1e-9)
                    total_score = delta[prev_t, i] + trans_score + dur_score + obs_score
                    
                    if total_score > best_prev_score:
                        best_prev_score = total_score
                        best_prev_state = i
                
                # Update Delta if this duration d is better than others for ending at t
                if best_prev_score > delta[t, j]:
                    delta[t, j] = best_prev_score
                    psi_state[t, j] = best_prev_state
                    psi_dur[t, j] = d

    # Termination & Backtracking
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

# --- 5. RUN AND COMPARE ---

predicted_states = hsmm_viterbi(observations, trans_mat, duration_probs, means, stds)

# Visualization in text format
print(f"{'Time':<5} | {'Obs (HR)':<10} | {'True':<6} | {'HSMM':<6}")
print("-" * 40)
for t in range(45, 56): # Inspect the noise spike area
    obs_str = f"{observations[t]:.1f}"
    is_spike = "<-- SPIKE" if t == 50 else ""
    print(f"{t:<5} | {obs_str:<10} | {states[true_states[t]]:<6} | {states[predicted_states[t]]:<6} {is_spike}")

print("\nAccuracy:", np.mean(predicted_states == true_states))
