import numpy as np
from hsmmlearn.hsmm import GaussianHSMM

obs_seq = np.array([
    1,1,3,2,2,2,1,5,1,5,4,1,1,1,2,2,5,5,1,5,
    2,5,4,2,4,5,3,1,2,4,8,8,7,7,8,6,6,9,6,8,
    8,10,8,6,9,10,6,9,6,10
])

obs_seq = np.array([1,1,3,2,2,2,1,5,1,5,4,1,1,1,2,2,5,5,1,5])

# Gaussian approximation of od_probs (unchanged across all JSONs)
means  = np.array([9.4830, 4.8910, 2.6710, 9.1430])  # Awake, Light, Deep, REM
scales = np.array([1.3586, 1.4308, 1.1352, 1.1994])

# rd_probs — shape (4 states, 4 max durations)  ← M=4 this time
durations = np.array([
    [0.03957, 0.12187, 0.29236, 0.54620],  # Awake  strongly skewed→long
    [0.11281, 0.17887, 0.27922, 0.42910],  # Light  skewed→long
    [0.10428, 0.17107, 0.27785, 0.44679],  # Deep   skewed→long
    [0.03668, 0.09773, 0.25020, 0.61539],  # REM    very strongly→long
])
# Transition matrix (zero diagonal — no self-loops)
tmat = np.array([
    [0.0, 0.9, 0.0, 0.1],  # Awake → Light, REM
    [0.1, 0.0, 0.5, 0.4],  # Light → Awake, Deep, REM
    [0.0, 1.0, 0.0, 0.0],  # Deep  → Light
    [0.2, 0.8, 0.0, 0.0],  # REM   → Awake, Light
])

# Emission Prob
means = np.array([0.0, 5.0, 10.0])
scales = np.ones_like(means)

hsmm = GaussianHSMM(
    means, scales, durations, tmat,
)

#observations, states = hsmm.sample(300)

decoded_states = hsmm.decode(obs_seq)

print(obs_seq)
print(decoded_states)