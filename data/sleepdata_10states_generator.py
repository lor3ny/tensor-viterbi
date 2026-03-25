import json
import random
import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1.  PARAMETERS – extended to 10 states
# =============================================================================

sleep_states = [
    "Awake",        # 0
    "Drowsy",       # 1
    "Light N1",     # 2
    "Light N2",     # 3
    "Deep N3",      # 4
    "Deep N4",      # 5
    "REM Early",    # 6
    "REM Mid",      # 7
    "REM Late",     # 8
    "Micro-Arousal",# 9
]
J = len(sleep_states)          # 10

sleep_emissions = np.arange(13)   # 13 HR bins (unchanged)
time_steps      = 1000000
max_duration    = 1000

# --- Observation sequence (unchanged logic) ----------------------------------
sleep_obs_seq = np.zeros(time_steps)
for i in range(time_steps):
    if i < 30:
        sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4])
    elif i < 70:
        sleep_obs_seq[i] = random.choice([5, 6, 7, 8, 9])
    else:
        sleep_obs_seq[i] = random.choice([0, 1, 2, 3, 4])

print("Generated Observations (Heart Rate):", sleep_obs_seq)

# --- Transition matrix (10 × 10) ---------------------------------------------
# Rows = from-state, Columns = to-state. Each row sums to 1.
#
# Physiological rationale:
#   Awake       → Drowsy (high)
#   Drowsy      → Light N1 or back to Awake
#   Light N1    → Light N2 or Drowsy
#   Light N2    → Deep N3 or Light N1
#   Deep N3     → Deep N4 or Light N2
#   Deep N4     → Deep N3 or Light N2
#   REM Early   → REM Mid or Light N2
#   REM Mid     → REM Late or REM Early
#   REM Late    → Light N2 or Awake
#   Micro-Arousal → Awake or Light N1

sleep_trans_mat = np.array([
#  Awk  Dro  LN1  LN2  DN3  DN4  RE   RM   RL   MA
  [0.00, 0.85, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10],  # Awake
  [0.15, 0.00, 0.70, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05],  # Drowsy
  [0.05, 0.20, 0.00, 0.60, 0.00, 0.00, 0.10, 0.00, 0.00, 0.05],  # Light N1
  [0.00, 0.05, 0.15, 0.00, 0.55, 0.00, 0.20, 0.00, 0.00, 0.05],  # Light N2
  [0.00, 0.00, 0.05, 0.20, 0.00, 0.70, 0.00, 0.00, 0.00, 0.05],  # Deep N3
  [0.00, 0.00, 0.00, 0.20, 0.75, 0.00, 0.00, 0.00, 0.00, 0.05],  # Deep N4
  [0.00, 0.00, 0.05, 0.10, 0.00, 0.00, 0.00, 0.80, 0.00, 0.05],  # REM Early
  [0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.15, 0.00, 0.75, 0.05],  # REM Mid
  [0.05, 0.00, 0.10, 0.65, 0.00, 0.00, 0.10, 0.05, 0.00, 0.05],  # REM Late
  [0.40, 0.10, 0.40, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Micro-Arousal
], dtype=float)

# Safety: re-normalise rows to absorb any rounding
sleep_trans_mat = (sleep_trans_mat.T / sleep_trans_mat.sum(axis=1)).T

# --- Emission probabilities (13 HR bins × 10 states) ------------------------
#
# Columns = states in the same order as sleep_states.
# Each column is later normalised to sum to 1.
#
# HR bin semantics (rough BPM mapping):
#   0-2  → very low HR  (deep sleep)
#   3-5  → low-medium   (deep / REM late)
#   6-8  → medium       (light sleep / REM)
#   9-11 → medium-high  (drowsy / micro-arousal)
#   12   → high HR      (awake / arousal)
#
#         Awk   Dro   LN1   LN2   DN3   DN4   RE    RM    RL    MA
sleep_emission_probs = np.array([
    [0.001, 0.001, 0.002, 0.002, 0.150, 0.200, 0.020, 0.010, 0.005, 0.001],  # bin 0
    [0.001, 0.002, 0.005, 0.010, 0.300, 0.350, 0.050, 0.030, 0.010, 0.002],  # bin 1
    [0.002, 0.005, 0.020, 0.050, 0.350, 0.300, 0.100, 0.080, 0.030, 0.005],  # bin 2
    [0.002, 0.010, 0.050, 0.100, 0.150, 0.120, 0.200, 0.150, 0.080, 0.010],  # bin 3
    [0.002, 0.020, 0.100, 0.200, 0.040, 0.030, 0.300, 0.300, 0.200, 0.020],  # bin 4
    [0.010, 0.050, 0.200, 0.300, 0.005, 0.000, 0.200, 0.250, 0.350, 0.050],  # bin 5
    [0.030, 0.100, 0.300, 0.200, 0.002, 0.000, 0.080, 0.120, 0.200, 0.100],  # bin 6
    [0.150, 0.250, 0.200, 0.100, 0.001, 0.000, 0.030, 0.040, 0.080, 0.200],  # bin 7
    [0.300, 0.300, 0.080, 0.030, 0.001, 0.000, 0.010, 0.010, 0.030, 0.250],  # bin 8
    [0.300, 0.150, 0.030, 0.005, 0.001, 0.000, 0.005, 0.005, 0.010, 0.200],  # bin 9
    [0.150, 0.080, 0.010, 0.002, 0.000, 0.000, 0.003, 0.003, 0.003, 0.100],  # bin 10
    [0.040, 0.020, 0.002, 0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.040],  # bin 11
    [0.012, 0.012, 0.001, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.022],  # bin 12
], dtype=float)

# --- Start probabilities (10 states) ----------------------------------------
sleep_start_probs = np.array([0.85, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

# --- Duration probabilities --------------------------------------------------
def gaussian_window(length, mean, std):
    x   = np.arange(length)
    g   = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return (g / g.sum()).tolist()

sleep_duration_probs = [
    gaussian_window(max_duration, mean=3,  std=1.5),  # Awake          – short bursts
    gaussian_window(max_duration, mean=4,  std=1.5),  # Drowsy         – brief
    gaussian_window(max_duration, mean=5,  std=2.0),  # Light N1
    gaussian_window(max_duration, mean=6,  std=2.0),  # Light N2
    gaussian_window(max_duration, mean=7,  std=2.0),  # Deep N3
    gaussian_window(max_duration, mean=8,  std=2.0),  # Deep N4        – longest deep
    gaussian_window(max_duration, mean=6,  std=2.0),  # REM Early
    gaussian_window(max_duration, mean=7,  std=2.0),  # REM Mid
    gaussian_window(max_duration, mean=8,  std=2.5),  # REM Late       – longest REM
    gaussian_window(max_duration, mean=2,  std=1.0),  # Micro-Arousal  – very short
]

# =============================================================================
# 2.  NORMALISE emission columns so each state sums to 1
# =============================================================================
col_sums     = sleep_emission_probs.sum(axis=0)
emission_norm = (sleep_emission_probs / col_sums).tolist()   # shape 13 × 10

emission_by_state = [
    [round(emission_norm[bin_][state], 6) for bin_ in range(len(sleep_emissions))]
    for state in range(J)
]

# =============================================================================
# 3.  BUILD JSON STRUCTURE
# =============================================================================
states_block = []
for i, name in enumerate(sleep_states):
    states_block.append({
        "id":             i,
        "name":           name,
        "emission_probs": emission_by_state[i],      # length 13
        "duration_probs": sleep_duration_probs[i],   # length max_duration
    })

config = {
    "seed":      42,
    "n_steps":   time_steps,
    "M":         max_duration,
    "emission":  "nonp",
    "duration":  "nonp",
    "obs_seq":   [int(v) + 1 for v in sleep_obs_seq.tolist()],
    "pi":        [round(float(p), 4) for p in sleep_start_probs],
    "trans_mat": [[round(float(v), 4) for v in row]
                  for row in sleep_trans_mat.tolist()],
    "n_bins":    int(len(sleep_emissions)),
    "states":    states_block,
}

# =============================================================================
# 4.  WRITE JSON
# =============================================================================
out_path = f"data/sleep_data_10states_{time_steps}_{max_duration}.json"
with open(out_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConfig written to: {out_path}")

# --- Quick validation printout -----------------------------------------------
print(f"\nStates ({J}): {sleep_states}")
print(f"emission / duration : {config['emission']} / {config['duration']}")
print(f"M                   : {config['M']}")
print(f"pi                  : {config['pi']}")
print("trans_mat rows (rounded to 2dp):")
for name, row in zip(sleep_states, config["trans_mat"]):
    print(f"  {name:<16}: {[round(v,2) for v in row]}")
print(f"Obs seq length      : {len(config['obs_seq'])}")
print(f"Emission pmf shape  : {J} states × {config['n_bins']} bins")
print(f"Duration pmf shape  : {J} states × {config['M']} steps")

# --- Row-sum sanity checks ---------------------------------------------------
print("\nRow-sum checks (should all be ~1.0):")
for name, row in zip(sleep_states, sleep_trans_mat):
    print(f"  {name:<16}: {row.sum():.6f}")
print("\nEmission column-sum checks (should all be ~1.0):")
for i, name in enumerate(sleep_states):
    print(f"  {name:<16}: {sum(emission_by_state[i]):.6f}")