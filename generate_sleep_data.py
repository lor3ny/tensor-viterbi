import json
import random
import numpy as np
 
# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)
 
# =============================================================================
# 1.  ORIGINAL PARAMETERS (unchanged from the provided code)
# =============================================================================
 
sleep_states    = ["Awake", "Light", "Deep", "REM"]
sleep_emissions = np.arange(13)
time_steps      = 300
max_duration    = 10
 
# --- Observation sequence ----------------------------------------------------
sleep_obs_seq = np.zeros(time_steps)
for i in range(time_steps):
    if i < 30:
        sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4])
    elif i < 70:
        sleep_obs_seq[i] = random.choice([5, 6, 7, 8, 9])
    else:
        sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4])
 
print("Generated Observations (Heart Rate):", sleep_obs_seq)
 
# --- Transition matrix -------------------------------------------------------
sleep_trans_mat = np.array([
    [0.0, 0.9, 0.0, 0.1],
    [0.1, 0.0, 0.5, 0.4],
    [0.0, 1.0, 0.0, 0.0],
    [0.2, 0.8, 0.0, 0.0],
])
 
# --- Emission probabilities (13 HR bins × 4 states) -------------------------
sleep_emission_probs = np.array([
    [0.001, 0.010, 0.150, 0.001],
    [0.001, 0.030, 0.300, 0.001],
    [0.002, 0.100, 0.350, 0.002],
    [0.002, 0.250, 0.150, 0.002],
    [0.002, 0.300, 0.040, 0.002],
    [0.010, 0.200, 0.005, 0.010],
    [0.030, 0.080, 0.002, 0.030],
    [0.150, 0.020, 0.001, 0.200],
    [0.300, 0.005, 0.001, 0.400],
    [0.300, 0.002, 0.001, 0.250],
    [0.150, 0.001, 0.000, 0.080],
    [0.040, 0.001, 0.000, 0.020],
    [0.012, 0.001, 0.000, 0.002],
])
 
# --- Start / duration probabilities ------------------------------------------
sleep_start_probs = np.array([0.9, 0.1, 0.0, 0.0])
 
def gaussian_window(length, mean, std):
    x = np.arange(length)
    g = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return (g / g.sum()).tolist()
 
sleep_duration_probs = [
    gaussian_window(max_duration, mean=5,  std=2),   # Awake
    gaussian_window(max_duration, mean=30, std=8),   # Light
    gaussian_window(max_duration, mean=50, std=10),  # Deep
    gaussian_window(max_duration, mean=25, std=5),   # REM
]
 
# =============================================================================
# 2.  NORMALISE columns of emission matrix so each state sums to 1
#     (hsmm nonparametric od requires a proper pmf per state)
# =============================================================================
col_sums = sleep_emission_probs.sum(axis=0)
emission_norm = (sleep_emission_probs / col_sums).tolist()   # shape 13 × 4
 
# Per-state emission vectors (list of 4 lists, each length 13)
emission_by_state = [
    [round(emission_norm[bin_][state], 6) for bin_ in range(len(sleep_emissions))]
    for state in range(len(sleep_states))
]
 
# =============================================================================
# 3.  BUILD JSON STRUCTURE
# =============================================================================
 
# states block — id, name, emission pmf, duration pmf
states_block = []
for i, name in enumerate(sleep_states):
    states_block.append({
        "id":            i,          # R is 1-indexed
        "name":          name,
        "emission_probs":      emission_by_state[i],      # length 13
        "duration_probs":      sleep_duration_probs[i],   # length max_duration
    })
 
config = {
    "seed":      42,
    "n_steps":   time_steps,
    "M":         max_duration,
    # hsmm nonparametric keyword for both distributions
    "emission":        "nonp",
    "duration":        "nonp",
    # observed sequence (1-indexed bins for R: add 1 to each value)
    "obs_seq":   [int(v) + 1 for v in sleep_obs_seq.tolist()],
    "pi":        [round(float(p), 4) for p in sleep_start_probs],
    "trans_mat":       [[round(float(v), 4) for v in row]
                  for row in sleep_trans_mat.tolist()],
    # number of HR bins
    "n_bins":    int(len(sleep_emissions)),
    "states":    states_block,
}
 
# =============================================================================
# 4.  WRITE JSON
# =============================================================================
 
out_path = "sleep_data.json"
with open(out_path, "w") as f:
    json.dump(config, f, indent=2)
 
print(f"\nConfig written to: {out_path}")
 
# --- Quick validation printout -----------------------------------------------
J = len(sleep_states)
print(f"\nStates ({J}): {sleep_states}")
print(f"emission / duration    : {config['emission']} / {config['duration']}")
print(f"M          : {config['M']}")
print(f"pi         : {config['pi']}")
print("trans_mat rows   :")
for row in config["trans_mat"]:
    print("  ", row)
print(f"Obs seq length : {len(config['obs_seq'])}")
print(f"Emission pmf shape : {J} states × {config['n_bins']} bins")
print(f"Duration pmf shape : {J} states × {config['M']} steps")