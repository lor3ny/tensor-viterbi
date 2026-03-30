import argparse
import csv
import os
import time

from tensor_viterbi import HSMM
from tensor_viterbi.viterbi import (
    decode_vanilla_viterbi,
    decode_log_tensor_viterbi_cached,
)

# ── ANSI colors (disabled automatically when output is not a terminal) ────────
import sys
_TTY = sys.stdout.isatty()
def _c(code): return code if _TTY else ""

R     = _c("\033[0m")
BOLD  = _c("\033[1m")
DIM   = _c("\033[2m")
CYAN  = _c("\033[96m")
GREEN = _c("\033[92m")
YEL   = _c("\033[93m")
GRAY  = _c("\033[90m")
WHITE = _c("\033[97m")
SEP   = GRAY + "─" * 52 + R
# ─────────────────────────────────────────────────────────────────────────────


def TIME_MEASURE(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"  {GRAY}time{R}  {WHITE}{func.__name__}{R}  {BOLD}{GREEN}{elapsed:.4f} s{R}\n")
    return result, elapsed


def TIME_BENCHMARK(func, *args, csv_path="benchmark.csv", iterations=100, **kwargs):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["function", "iteration", "elapsed_s"])
        for i, t in enumerate(times):
            writer.writerow([func.__name__, i, f"{t:.6f}"])

    avg, mn, mx = sum(times) / len(times), min(times), max(times)
    print(f"  {WHITE}{func.__name__}{R}")
    print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend")
    parser.add_argument("--cpp", action="store_true", help="Enable C++ backend")
    parser.add_argument("--baseline", action="store_true", help="Enable baselines (HSMMLearn C++, Vanilla Viterbi)")
    parser.add_argument("--mode", "-m", choices=["validate", "measure", "benchmark"], required=True)
    parser.add_argument("--data-path", "-dp", type=str, default="data/3states_20steps_4dur.json")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        parser.error(f"data_path '{args.data_path}' does not exist")

    data_path = args.data_path

    if args.cpp:
        from tensor_viterbi.viterbi import (
            decode_tensor_viterbi_cpp,
        )
    if args.cuda:
        from tensor_viterbi.viterbi import (
            decode_tensor_viterbi_cuda,
        )
    if args.baseline:
        from validation.hsmmlearn_viterbi import (
            validate,
            measure_baseline,
            benchmark_baseline
        )
        from validation.hsmmlearn_py_viterbi import (
            validate_py,
            measure_baseline_py,
            benchmark_baseline_py
        )


    my_hsmm = HSMM.load_model(data_path)

    N = len(my_hsmm.states)
    T = len(my_hsmm.obs_seq)
    D = my_hsmm.duration_probs.shape[0]

    # Args for the native (C++/CUDA) functions: (n_states, trans_mat, emission_probs, start_probs, duration_probs, obs_seq)
    _cpp_args = (N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)

    print(f"\n{SEP}")
    print(f"  {BOLD}{CYAN}DATA SUMMARY{R}")
    print(SEP)
    print(f"  {GRAY}data path{R}  {WHITE}{data_path}{R}")
    print(f"  {GRAY}states (N){R}  {BOLD}{N}{R}  {DIM}{my_hsmm.states}{R}")
    print(f"  {GRAY}steps  (T){R}  {BOLD}{T}{R}")
    print(f"  {GRAY}max dur(D){R}  {BOLD}{D}{R}")
    print(f"{SEP}\n")

    if args.mode == "validate":

        print(f"{YEL}{BOLD}▶ Tensor Viterbi (Cached){R}")
        tc_predicted_states = decode_log_tensor_viterbi_cached(my_hsmm)
        validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)

        # v_predicted_states = decode_vanilla_viterbi(my_hsmm)
        # validate("Vanilla vs Baseline", v_predicted_states, data_path)

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            cpp_predicted_states = decode_tensor_viterbi_cpp(*_cpp_args)
            validate("C++ vs Baseline", cpp_predicted_states, data_path)

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            cuda_predicted_states = decode_tensor_viterbi_cuda(*_cpp_args)
            validate("CUDA vs Baseline", cuda_predicted_states, data_path)


    elif args.mode == "measure":
        baseline_elapsed = None
        if args.baseline:
            print(f"{YEL}{BOLD}▶ HSMMLearn C++ (baseline){R}")
            baseline_elapsed = measure_baseline(data_path)

            # TIME_MEASURE(decode_vanilla_viterbi, my_hsmm)
            # measure_baseline_py(data_path)

        print(f"{YEL}{BOLD}▶ Tensor Viterbi (Cached){R}")
        _, tc_elapsed = TIME_MEASURE(decode_log_tensor_viterbi_cached, my_hsmm)
        if baseline_elapsed is not None:
            print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / tc_elapsed:.2f}x{R} vs HSMMLearn C++\n")

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            _, cpp_elapsed = TIME_MEASURE(decode_tensor_viterbi_cpp, *_cpp_args)
            if baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / cpp_elapsed:.2f}x{R} vs HSMMLearn C++\n")

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            _, cuda_elapsed = TIME_MEASURE(decode_tensor_viterbi_cuda, *_cpp_args)
            if baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / cuda_elapsed:.2f}x{R} vs HSMMLearn C++\n")


    elif args.mode == "benchmark":

        # TIME_BENCHMARK(decode_log_tensor_viterbi_cached, my_hsmm, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            TIME_BENCHMARK(decode_tensor_viterbi_cpp, *_cpp_args, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            TIME_BENCHMARK(decode_tensor_viterbi_cuda, *_cpp_args, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.baseline:
            print(f"{YEL}{BOLD}▶ HSMMLearn C++ (baseline){R}")
            benchmark_baseline(data_path, csv_path="viterbi_benchmark.csv", iterations=10)

            print(f"{YEL}{BOLD}▶ HSMMLearn Python (baseline){R}")
            benchmark_baseline_py(data_path, csv_path="viterbi_benchmark.csv", iterations=10)

            print(f"{YEL}{BOLD}▶ Vanilla Viterbi (baseline){R}")
            TIME_BENCHMARK(decode_vanilla_viterbi, my_hsmm, csv_path="viterbi_benchmark.csv", iterations=10)
