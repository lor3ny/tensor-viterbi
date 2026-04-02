import argparse
import csv
import os
import time
import numpy as np

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



def compute_accuracy(true_states, predicted_states):
    true_states = np.array(true_states)
    predicted_states = np.array(predicted_states)
    return np.sum(true_states == predicted_states) / len(true_states)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--py", action="store_true", help="Enable Python backend")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend")
    parser.add_argument("--cpp", action="store_true", help="Enable C++ backend")
    parser.add_argument("--omp", action="store_true", help="Enable OpenMP backend")
    parser.add_argument("--baseline",     action="store_true", help="Enable all baselines (HSMMLearn C++ + OMP)")
    parser.add_argument("--baseline-cpp", action="store_true", help="Enable HSMMLearn C++ baseline only")
    parser.add_argument("--baseline-omp", action="store_true", help="Enable HSMMLearn OMP baseline only")
    parser.add_argument("--mode", "-m", choices=["validate", "measure", "benchmark"], required=True)
    parser.add_argument("--system", "-sys", required=False, default="leonardo", help="System name (for benchmark CSV naming)")
    parser.add_argument("--iterations", "-it", type=int, default=10, help="Number of benchmark iterations")
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
    if args.omp:
        from tensor_viterbi.viterbi import (
            decode_tensor_viterbi_omp,
        )
    _run_baseline_cpp = args.baseline or args.baseline_cpp
    _run_baseline_omp = args.baseline or args.baseline_omp
    if _run_baseline_cpp:
        from validation.hsmmlearn_viterbi import (
            validate,
            measure_baseline,
            benchmark_baseline
        )
    if _run_baseline_omp:
        from validation.hsmmlearn_omp_viterbi import (
            validate as validate_omp,
            measure_baseline as measure_baseline_omp,
            benchmark_baseline as benchmark_baseline_omp,
        )
    if _run_baseline_cpp or _run_baseline_omp:
        from validation.hsmmlearn_py_viterbi import (
            validate_py,
            measure_baseline_py,
            benchmark_baseline_py
        )


    my_hsmm = HSMM.load_model(data_path)

    N = len(my_hsmm.states)
    T = len(my_hsmm.obs_seq)
    D = my_hsmm.duration_probs.shape[0]


    print(f"\n{SEP}")
    print(f"  {BOLD}{CYAN}DATA SUMMARY{R}")
    print(SEP)
    print(f"  {GRAY}data path{R}  {WHITE}{data_path}{R}")
    print(f"  {GRAY}states (N){R}  {BOLD}{N}{R}  {DIM}{my_hsmm.states}{R}")
    print(f"  {GRAY}steps  (T){R}  {BOLD}{T}{R}")
    print(f"  {GRAY}max dur(D){R}  {BOLD}{D}{R}")
    print(f"{SEP}\n")

    if args.mode == "validate":

        my_hsmm.to_log_space()

        if args.py:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi (Cached){R}")
            tc_predicted_states = decode_log_tensor_viterbi_cached(my_hsmm)
            validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)

        # v_predicted_states = decode_vanilla_viterbi(my_hsmm)
        # validate("Vanilla vs Baseline", v_predicted_states, data_path)

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            cpp_predicted_states = decode_tensor_viterbi_cpp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            validate("C++ vs Baseline", cpp_predicted_states, data_path)

        if args.omp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi OMP{R}")
            omp_predicted_states = decode_tensor_viterbi_omp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            validate("OMP vs Baseline", omp_predicted_states, data_path)
            validate_omp("OMP vs Baseline (HSMMLearn OMP)", omp_predicted_states, data_path)

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            cuda_predicted_states = decode_tensor_viterbi_cuda(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            validate("CUDA vs Baseline", cuda_predicted_states, data_path)


    elif args.mode == "measure":
        baseline_elapsed = None
        omp_baseline_elapsed = None
        cpp_elapsed = None
        if _run_baseline_cpp:
            print(f"{YEL}{BOLD}▶ HSMMLearn C++ (baseline){R}")
            baseline_elapsed = measure_baseline(data_path)

        if _run_baseline_omp:
            print(f"{YEL}{BOLD}▶ HSMMLearn OMP (baseline){R}")
            omp_baseline_elapsed = measure_baseline_omp(data_path)

            # TIME_MEASURE(decode_vanilla_viterbi, my_hsmm)
            # measure_baseline_py(data_path)
    

        if args.py:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi (Cached){R}")
            my_hsmm = HSMM.load_model(data_path)
            def _py_cached():
                my_hsmm.to_log_space()
                return decode_log_tensor_viterbi_cached(my_hsmm)
            _py_cached.__name__ = decode_log_tensor_viterbi_cached.__name__
            _, tc_elapsed = TIME_MEASURE(_py_cached)
            validate("Validate", _, data_path)
            if baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / tc_elapsed:.2f}x{R} vs HSMMLearn C++\n")

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            my_hsmm = HSMM.load_model(data_path)
            def _cpp():
                my_hsmm.to_log_space()
                return decode_tensor_viterbi_cpp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            _cpp.__name__ = decode_tensor_viterbi_cpp.__name__
            _, cpp_elapsed = TIME_MEASURE(_cpp)
            validate("Validate", _, data_path)
            if baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / cpp_elapsed:.2f}x{R} vs HSMMLearn C++\n")

        if args.omp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi OMP{R}")
            my_hsmm = HSMM.load_model(data_path)
            def _omp():
                my_hsmm.to_log_space()
                return decode_tensor_viterbi_omp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            _omp.__name__ = decode_tensor_viterbi_omp.__name__
            _, omp_elapsed = TIME_MEASURE(_omp)
            validate("Validate", _, data_path)
            if omp_baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{omp_baseline_elapsed / omp_elapsed:.2f}x{R} vs HSMMLearn OMP C++\n")
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{cpp_elapsed / omp_elapsed:.2f}x{R} vs Tensor Viterbi C++\n")

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            my_hsmm = HSMM.load_model(data_path)
            def _cuda():
                my_hsmm.to_log_space()
                return decode_tensor_viterbi_cuda(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
            _cuda.__name__ = decode_tensor_viterbi_cuda.__name__
            _, cuda_elapsed = TIME_MEASURE(_cuda)
            validate("Validate", _, data_path)
            if baseline_elapsed is not None:
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{baseline_elapsed / cuda_elapsed:.2f}x{R} vs HSMMLearn C++\n")
                print(f"  {GRAY}speedup{R}  {BOLD}{GREEN}{cpp_elapsed / cuda_elapsed:.2f}x{R} vs Tensor Viterbi C++\n")



    elif args.mode == "benchmark":
        os.makedirs(f"results/{args.system}", exist_ok=True)
        _csv = os.path.join(f"results/{args.system}", f"{N}s_{D}d_{T}t.csv")
        _stem, _ext = os.path.splitext(_csv)

        if args.py:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi Python{R}")
            _fname = decode_log_tensor_viterbi_cached.__name__
            _times = []
            for _ in range(args.iterations):
                my_hsmm = HSMM.load_model(data_path)
                start = time.perf_counter()
                my_hsmm.to_log_space()
                decode_log_tensor_viterbi_cached(my_hsmm)
                _times.append(time.perf_counter() - start)
            with open(f"{_stem}_{_fname}{_ext}", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
                for i, t in enumerate(_times):
                    writer.writerow([_fname, N, T, D, i, f"{t:.6f}"])
            avg, mn, mx = sum(_times) / len(_times), min(_times), max(_times)
            print(f"  {WHITE}{_fname}{R}")
            print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")

        if args.cpp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi C++{R}")
            _fname = decode_tensor_viterbi_cpp.__name__
            _times = []
            for _ in range(args.iterations):
                my_hsmm = HSMM.load_model(data_path)
                start = time.perf_counter()
                my_hsmm.to_log_space()
                decode_tensor_viterbi_cpp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
                _times.append(time.perf_counter() - start)
            with open(f"{_stem}_{_fname}{_ext}", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
                for i, t in enumerate(_times):
                    writer.writerow([_fname, N, T, D, i, f"{t:.6f}"])
            avg, mn, mx = sum(_times) / len(_times), min(_times), max(_times)
            print(f"  {WHITE}{_fname}{R}")
            print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")

        if args.omp:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi OMP{R}")
            _fname = decode_tensor_viterbi_omp.__name__
            _times = []
            for _ in range(args.iterations):
                my_hsmm = HSMM.load_model(data_path)
                start = time.perf_counter()
                my_hsmm.to_log_space()
                decode_tensor_viterbi_omp(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
                _times.append(time.perf_counter() - start)
            with open(f"{_stem}_{_fname}{_ext}", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
                for i, t in enumerate(_times):
                    writer.writerow([_fname, N, T, D, i, f"{t:.6f}"])
            avg, mn, mx = sum(_times) / len(_times), min(_times), max(_times)
            print(f"  {WHITE}{_fname}{R}")
            print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")

        if args.cuda:
            print(f"{YEL}{BOLD}▶ Tensor Viterbi CUDA{R}")
            _fname = decode_tensor_viterbi_cuda.__name__
            _times = []
            for _ in range(args.iterations):
                my_hsmm = HSMM.load_model(data_path)
                start = time.perf_counter()
                my_hsmm.to_log_space()
                decode_tensor_viterbi_cuda(N, my_hsmm.trans_mat, my_hsmm.emission_probs, my_hsmm.duration_probs_linear, my_hsmm.start_probs, my_hsmm.duration_probs, my_hsmm.obs_seq)
                _times.append(time.perf_counter() - start)
            with open(f"{_stem}_{_fname}{_ext}", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
                for i, t in enumerate(_times):
                    writer.writerow([_fname, N, T, D, i, f"{t:.6f}"])
            avg, mn, mx = sum(_times) / len(_times), min(_times), max(_times)
            print(f"  {WHITE}{_fname}{R}")
            print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")

        if _run_baseline_cpp:
            print(f"{YEL}{BOLD}▶ HSMMLearn C++ (baseline){R}")
            _csv_base = os.path.splitext(_csv)[0]
            benchmark_baseline(data_path, csv_path=f"{_csv_base}_HSMMLearn_CPP.csv", iterations=args.iterations, n_states=N, timesteps=T, max_duration=D)

        if _run_baseline_omp:
            print(f"{YEL}{BOLD}▶ HSMMLearn OMP (baseline){R}")
            _csv_base = os.path.splitext(_csv)[0]
            benchmark_baseline_omp(data_path, csv_path=f"{_csv_base}_HSMMLearn_OMP.csv", iterations=args.iterations, n_states=N, timesteps=T, max_duration=D)

            # print(f"{YEL}{BOLD}▶ HSMMLearn Python (baseline){R}")
            # benchmark_baseline_py(data_path, csv_path=_csv, iterations=args.iterations)

            # print(f"{YEL}{BOLD}▶ Vanilla Viterbi (baseline){R}")
            # TIME_BENCHMARK(decode_vanilla_viterbi, my_hsmm, **_bkw)
