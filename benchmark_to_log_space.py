import os
import sys
import time

os.environ["SYS_NAME"] = "xeon8480/gnu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor_viterbi import HSMM

STATES     = [10, 15, 25, 50, 75]
DURATIONS  = [100, 250, 500, 1000]
TIMESTEPS  = [1000, 10000, 100000, 1000000]
ITERATIONS = 5
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

print(f"{'States':>8}  {'Duration':>10}  {'Timesteps':>12}  {'Iter':>5}  {'Time (s)':>12}")
print("-" * 58)

summary = []  # (n_states, duration, timesteps, avg)

for n_states in STATES:
    for duration in DURATIONS:
        for timesteps in TIMESTEPS:
            path = os.path.join(DATA_DIR, f"{n_states}states_{timesteps}steps_{duration}dur.json")
            if not os.path.exists(path):
                print(f"{n_states:>8}  {duration:>10}  {timesteps:>12}  {'N/A':>5}  {'FILE NOT FOUND':>12}")
                continue

            times = []
            for i in range(ITERATIONS):
                my_hsmm = HSMM.load_model(path)
                start = time.perf_counter()
                my_hsmm.to_log_space()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"{n_states:>8}  {duration:>10}  {timesteps:>12}  {i+1:>5}  {elapsed:>12.6f}")
            avg = sum(times) / len(times)
            print(f"{'':>8}  {'':>10}  {'':>12}  {'avg':>5}  {avg:>12.6f}")
            print()
            summary.append((n_states, duration, timesteps, avg))

print("=" * 58)
print("SUMMARY OF AVERAGES")
print("=" * 58)
print(f"{'States':>8}  {'Duration':>10}  {'Timesteps':>12}  {'Avg (s)':>12}")
print("-" * 58)
for n_states, duration, timesteps, avg in summary:
    print(f"{n_states:>8}  {duration:>10}  {timesteps:>12}  {avg:>12.6f}")
