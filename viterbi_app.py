#!/usr/bin/env python3
"""
viterbi_app.py — benchmark tensor-viterbi backends and validate results.

Invoked by run_benchmark.py (via SLURM or locally).

Usage:
  python viterbi_app.py --system <sys> --toolchain <tc> --cpp --omp \
      --iterations 6 --data-path data/10states_1000000steps_100dur.json
"""

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Set SYS_NAME before tensor_viterbi imports — native.py reads it at module load time.
if "SYS_NAME" not in os.environ:
    _argv = sys.argv[1:]
    _pairs = list(zip(_argv, _argv[1:]))
    _sys = next((v for a, v in _pairs if a in ("--system", "-sys")), None)
    _tc  = next((v for a, v in _pairs if a in ("--toolchain", "-tc")), None)
    if _sys and _tc:
        os.environ["SYS_NAME"] = f"{_sys}/{_tc}"

from tensor_viterbi import HSMM
from tensor_viterbi.metrics import get_collector
from tensor_viterbi.viterbi import decode_log_tensor_viterbi_cached
import tensor_viterbi.viterbi.native as _native_mod

SCRIPT_DIR = Path(__file__).resolve().parent

_TTY  = sys.stdout.isatty()
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


def _validate(result, ref_path: Path) -> None:
    if result is None or not ref_path.exists():
        return
    ref = np.load(str(ref_path))
    acc = float(np.mean(np.asarray(result) == np.asarray(ref)))
    color = GREEN if acc >= 0.95 else YEL
    print(f"  {GRAY}validation{R}  {color}{BOLD}{acc:.2%} match{R}\n")


def _bench(label, fname, run_fn, my_hsmm, N, T, D, iterations, stem, ext, collector):
    """Run run_fn for `iterations` iterations, time each, write CSV, return last result."""
    print(f"{YEL}{BOLD}▶ {label}{R}")
    times = []
    last_result = None
    collector.start()
    with open(f"{stem}_{fname}{ext}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
        for i in range(iterations):
            tmp = copy.copy(my_hsmm)
            t0 = time.perf_counter()
            result = run_fn(tmp)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            w.writerow([fname, N, T, D, i, f"{elapsed:.6f}"])
            f.flush()
            if i == iterations - 1:
                last_result = result
    metrics = collector.stop()
    if collector.column_names():
        with open(f"{stem}_{fname}_metrics{ext}", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["function", "n_states", "timesteps", "max_duration",
                        "total_iterations", *collector.column_names()])
            w.writerow([fname, N, T, D, iterations,
                        *[metrics.get(k, "") for k in collector.column_names()]])
    avg, mn, mx = sum(times) / len(times), min(times), max(times)
    print(f"  {WHITE}{fname}{R}")
    print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")
    return last_result


def _bench_baseline(label, fname, model, obs_seq, N, T, D, iterations, stem, ext, collector):
    """Like _bench but for hsmmlearn models. Returns decoded states from last iteration."""
    print(f"{YEL}{BOLD}▶ {label}{R}")
    times = []
    last_result = None
    collector.start()
    with open(f"{stem}_{fname}{ext}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "n_states", "timesteps", "max_duration", "iteration", "elapsed_s"])
        for i in range(iterations):
            t0 = time.perf_counter()
            result = model.decode(obs_seq)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            w.writerow([fname, N, T, D, i, f"{elapsed:.6f}"])
            f.flush()
            if i == iterations - 1:
                last_result = result
    metrics = collector.stop()
    if collector.column_names():
        with open(f"{stem}_{fname}_metrics{ext}", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["function", "n_states", "timesteps", "max_duration",
                        "total_iterations", *collector.column_names()])
            w.writerow([fname, N, T, D, iterations,
                        *[metrics.get(k, "") for k in collector.column_names()]])
    avg, mn, mx = sum(times) / len(times), min(times), max(times)
    print(f"  {WHITE}{fname}{R}")
    print(f"  avg {BOLD}{GREEN}{avg:.4f} s{R}   min {GREEN}{mn:.4f} s{R}   max {GREEN}{mx:.4f} s{R}\n")
    return last_result


def main():
    parser = argparse.ArgumentParser(description="Benchmark tensor-viterbi backends.")
    parser.add_argument("--system",       "-sys", required=True)
    parser.add_argument("--toolchain",    "-tc",  required=True)
    parser.add_argument("--py",           action="store_true")
    parser.add_argument("--cpp",          action="store_true")
    parser.add_argument("--omp",          action="store_true")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--baseline",     action="store_true", help="Enable all baselines")
    parser.add_argument("--baseline-cpp", action="store_true", dest="baseline_cpp")
    parser.add_argument("--baseline-omp", action="store_true", dest="baseline_omp")
    parser.add_argument("--iterations",   type=int, default=6)
    parser.add_argument("--data-path",    "-dp", required=True)
    args = parser.parse_args()

    _native_mod.configure(args.system, args.toolchain)

    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: data_path '{data_path}' does not exist.")
        sys.exit(1)

    _run_baseline_cpp = args.baseline or args.baseline_cpp
    _run_baseline_omp = args.baseline or args.baseline_omp

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

    _metrics_backend = os.environ.get("SYS_METRICS_BACKEND", "").strip()
    collector = get_collector(_metrics_backend or None)

    sys_name = f"{args.system}/{args.toolchain}"
    results_dir = SCRIPT_DIR / "results" / sys_name
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = str(results_dir / f"{N}s_{D}d_{T}t")
    ext  = ".csv"

    ref_path = results_dir / (Path(data_path).stem + "_reference.npy")

    bkw = dict(N=N, T=T, D=D, iterations=args.iterations,
               stem=stem, ext=ext, collector=collector)

    # ── Baselines (produce the reference .npy from last iteration) ────────────
    if _run_baseline_cpp:
        from validation.hsmmlearn_viterbi import load_sleep_model_hsmmlearn
        model, obs_seq = load_sleep_model_hsmmlearn(data_path)
        ref_states = _bench_baseline("HSMMLearn C++ (baseline)", "HSMMLearn_CPP",
                                     model, obs_seq, **bkw)
        np.save(str(ref_path), ref_states)
        print(f"  {GRAY}reference{R}  saved → {WHITE}{ref_path}{R}\n")

    if _run_baseline_omp:
        from validation.hsmmlearn_omp_viterbi import load_sleep_model_hsmmlearn as load_omp
        model_omp, obs_seq_omp = load_omp(data_path)
        ref_states_omp = _bench_baseline("HSMMLearn OMP (baseline)", "HSMMLearn_OMP",
                                         model_omp, obs_seq_omp, **bkw)
        if not ref_path.exists():
            np.save(str(ref_path), ref_states_omp)
            print(f"  {GRAY}reference{R}  saved → {WHITE}{ref_path}{R}\n")

    # ── Tensor Viterbi backends ───────────────────────────────────────────────
    if args.py:
        def _run(h): h.to_log_space(); return decode_log_tensor_viterbi_cached(h)
        res = _bench("Tensor Viterbi Python", decode_log_tensor_viterbi_cached.__name__,
                     _run, my_hsmm, **bkw)
        _validate(res, ref_path)

    if args.cpp:
        from tensor_viterbi.viterbi import decode_tensor_viterbi_cpp
        def _run(h):
            h.to_log_space()
            return decode_tensor_viterbi_cpp(N, h.trans_mat, h.emission_probs,
                                             h.duration_probs_linear, h.start_probs,
                                             h.duration_probs, h.obs_seq)
        res = _bench("Tensor Viterbi C++", decode_tensor_viterbi_cpp.__name__,
                     _run, my_hsmm, **bkw)
        _validate(res, ref_path)

    if args.omp:
        from tensor_viterbi.viterbi import decode_tensor_viterbi_omp
        def _run(h):
            h.to_log_space()
            return decode_tensor_viterbi_omp(N, h.trans_mat, h.emission_probs,
                                             h.duration_probs_linear, h.start_probs,
                                             h.duration_probs, h.obs_seq)
        res = _bench("Tensor Viterbi OMP", decode_tensor_viterbi_omp.__name__,
                     _run, my_hsmm, **bkw)
        _validate(res, ref_path)

    if args.gpu:
        from tensor_viterbi.viterbi import decode_tensor_viterbi_cuda
        def _run(h):
            h.to_log_space()
            return decode_tensor_viterbi_cuda(N, h.trans_mat, h.emission_probs,
                                              h.duration_probs_linear, h.start_probs,
                                              h.duration_probs, h.obs_seq)
        res = _bench("Tensor Viterbi CUDA", decode_tensor_viterbi_cuda.__name__,
                     _run, my_hsmm, **bkw)
        _validate(res, ref_path)

    ref_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
