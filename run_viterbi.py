import argparse
import csv
import os
import time

from tensor_viterbi import HSMM
from tensor_viterbi.viterbi import (
    decode_vanilla_viterbi,
    decode_log_tensor_viterbi_cached,
    decode_tensor_viterbi_cpp,
    decode_tensor_viterbi_cuda,  # uncomment when CUDA is available
)

from validation.hsmmlearn_viterbi import validate, measure_baseline, benchmark_baseline
from validation.hsmmlearn_py_viterbi import validate_py, measure_baseline_py, benchmark_baseline_py


def TIME_MEASURE(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"\nExecution time of {func.__name__}: {elapsed:.4f} seconds\n")
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

    print(f"\n{func.__name__}: avg={sum(times)/len(times):.4f}s  min={min(times):.4f}s  max={max(times):.4f}s\n")


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

    
    
    my_hsmm = HSMM.load_model(data_path)

    N = len(my_hsmm.states)
    T = len(my_hsmm.obs_seq)
    D = my_hsmm.duration_probs.shape[0]
    print(f"\n===== DATA SUMMARY =====")
    print(f"  Data path : {data_path}")
    print(f"  States (N): {N}  -> {my_hsmm.states}")
    print(f"  Steps  (T): {T}")
    print(f"  Max dur(D): {D}")
    print(f"========================\n")

    if args.mode == "validate":

        print("\nRunning: Tensor Viterbi (Cached)...\n")
        tc_predicted_states = decode_log_tensor_viterbi_cached(my_hsmm)
        validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)

        if args.cpp:
            print("\nRunning: Tensor Viterbi C++...\n")
            cpp_predicted_states = decode_tensor_viterbi_cpp(data_path)
            validate("C++ vs Baseline", cpp_predicted_states, data_path)

        if args.cuda:
            print("\nRunning: Tensor Viterbi CUDA...\n")
            cuda_predicted_states = decode_tensor_viterbi_cuda(data_path)
            validate("CUDA vs Baseline", cuda_predicted_states, data_path)

        if args.baseline:
            print("\nRunning: Vanilla Viterbi (baseline)...\n")
            v_predicted_states = decode_vanilla_viterbi(my_hsmm)
            validate("Vanilla vs Baseline", v_predicted_states, data_path)


    elif args.mode == "measure":
        baseline_elapsed = None
        if args.baseline:
            print("\nRunning: HSMMLearn C++ (baseline)...\n")
            baseline_elapsed = measure_baseline(data_path)

            # print("Running: Vanilla Viterbi (baseline)...")
            # TIME_MEASURE(decode_vanilla_viterbi, my_hsmm)

            # print("Running: HSMMLearn Python (baseline)...")
            # measure_baseline_py(data_path)


        #print("\nRunning: Tensor Viterbi (Cached)...\n")
        #_, tc_elapsed = TIME_MEASURE(decode_log_tensor_viterbi_cached, my_hsmm)
        #if baseline_elapsed is not None:
        #    print(f"\n  -> speedup vs HSMMLearn C++: {baseline_elapsed / tc_elapsed:.2f}x\n")

        if args.cpp:
            print("\nRunning: Tensor Viterbi C++...\n")
            _, cpp_elapsed = TIME_MEASURE(decode_tensor_viterbi_cpp, data_path)
            if baseline_elapsed is not None:
                print(f"\n  -> speedup vs HSMMLearn C++: {baseline_elapsed / cpp_elapsed:.2f}x\n")

        if args.cuda:
            print("\nRunning: Tensor Viterbi CUDA...\n")
            _, cuda_elapsed = TIME_MEASURE(decode_tensor_viterbi_cuda, data_path)
            if baseline_elapsed is not None:
                print(f"\n  -> speedup vs HSMMLearn C++: {baseline_elapsed / cuda_elapsed:.2f}x\n")


    elif args.mode == "benchmark":

        #print("\nRunning: Tensor Viterbi (Cached)...\n")
        #TIME_BENCHMARK(decode_log_tensor_viterbi_cached, my_hsmm, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.cpp:
            print("\nRunning: Tensor Viterbi C++...\n")
            TIME_BENCHMARK(decode_tensor_viterbi_cpp, data_path, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.cuda:
            print("\nRunning: Tensor Viterbi CUDA...\n")
            TIME_BENCHMARK(decode_tensor_viterbi_cuda, data_path, csv_path="viterbi_benchmark.csv", iterations=10)

        if args.baseline:
            print("\nRunning: HSMMLearn C++ (baseline)...\n")
            benchmark_baseline(data_path, csv_path="viterbi_benchmark.csv", iterations=10)

            print("\nRunning: HSMMLearn Python (baseline)...\n")
            benchmark_baseline_py(data_path, csv_path="viterbi_benchmark.csv", iterations=10)

            print("\nRunning: Vanilla Viterbi (baseline)...\n")
            TIME_BENCHMARK(decode_vanilla_viterbi, my_hsmm, csv_path="viterbi_benchmark.csv", iterations=10)
