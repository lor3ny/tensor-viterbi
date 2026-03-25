import argparse
import csv
import os
import time

from tensor_viterbi import HSMM
from tensor_viterbi.viterbi import (
    decode_vanilla_viterbi,
    decode_log_tensor_viterbi_cached,
    decode_tensor_viterbi_cpp,
    # decode_tensor_viterbi_cuda,  # uncomment when CUDA is available
)

from validation.hsmmlearn_viterbi import validate, measure_baseline, benchmark_baseline


def TIME_MEASURE(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"Execution time of {func.__name__}: {elapsed:.4f} seconds")
    return result


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

    print(f"{func.__name__}: avg={sum(times)/len(times):.4f}s  min={min(times):.4f}s  max={max(times):.4f}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", choices=["validate", "measure", "benchmark"], required=True)
    parser.add_argument("--data-path", "-dp", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        parser.error(f"data_path '{args.data_path}' does not exist")

    data_path = args.data_path

    hsmm_sleep = HSMM.load_model(data_path)

    if args.mode == "validate":
        v_predicted_states  = decode_vanilla_viterbi(hsmm_sleep)
        tc_predicted_states = decode_log_tensor_viterbi_cached(hsmm_sleep)
        cpp_predicted_states  = decode_tensor_viterbi_cpp(data_path)
        #cuda_predicted_states = decode_tensor_viterbi_cuda(data_path)
        validate("Vanilla vs Baseline", v_predicted_states, data_path)
        validate("Tensor (Cached) vs Baseline", tc_predicted_states, data_path)
        validate("C++ vs Baseline", cpp_predicted_states, data_path)
        #validate("CUDA vs Baseline", cuda_predicted_states, data_path)

    elif args.mode == "measure":
        TIME_MEASURE(decode_vanilla_viterbi, hsmm_sleep)
        TIME_MEASURE(decode_log_tensor_viterbi_cached, hsmm_sleep)
        TIME_MEASURE(decode_tensor_viterbi_cpp, data_path)
        #TIME_MEASURE(decode_tensor_viterbi_cuda, data_path)
        measure_baseline(data_path)

    elif args.mode == "benchmark":
        TIME_BENCHMARK(decode_vanilla_viterbi, hsmm_sleep, csv_path="viterbi_benchmark.csv", iterations=10)
        TIME_BENCHMARK(decode_log_tensor_viterbi_cached, hsmm_sleep, csv_path="viterbi_benchmark.csv", iterations=10)
        TIME_BENCHMARK(decode_tensor_viterbi_cpp, data_path, csv_path="viterbi_benchmark.csv", iterations=10)
        #TIME_BENCHMARK(decode_tensor_viterbi_cuda, data_path, csv_path="viterbi_benchmark.csv", iterations=10)
        benchmark_baseline(data_path, csv_path="viterbi_benchmark.csv", iterations=10)
