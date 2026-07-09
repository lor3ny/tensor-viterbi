"""Backend-flag selection (--py/--cpp/--omp/--gpu/--baseline*) shared by
plan and likwid, plus the flag -> output-CSV-stem mapping used by the
resume/completeness check in execution.py. Mirrors viterbi_app.py exactly —
do not change viterbi_app.py's flag names/semantics without updating this.
"""

from __future__ import annotations

BACKEND_FLAGS = ["py", "cpp", "omp", "gpu", "baseline", "baseline-cpp", "baseline-omp"]

LIKWID_CPU_FLAGS = ["--baseline", "--baseline-omp", "--cpp", "--omp"]
LIKWID_PERF_GROUPS = ["FLOPS_DP", "MEM", "L3", "L2", "BRANCH", "TMA"]

# Pre-existing discrepancy in the old run_benchmark.py, preserved exactly:
# run_likwid_local() used a 10000-step/1000-duration file, while
# generate_likwid_slrm()'s heredoc hardcoded a different 1000-step/500-duration
# file. Since "measured behavior must remain byte-identical to today" for
# each backend, both defaults are kept rather than unified.
LIKWID_DATA_LOCAL = "data/75states_10000steps_1000dur.json"
LIKWID_DATA_SLURM = "data/75states_1000steps_500dur.json"

# flag name -> the `fname` viterbi_app.py's _bench()/_bench_baseline() uses,
# i.e. the "<stem>_<fname>.csv" file it writes on success. "baseline" expands
# to both HSMMLearn CSVs since viterbi_app.py runs both when --baseline is set.
_FNAME_FOR_FLAG = {
    "py":           ["decode_log_tensor_viterbi_cached"],
    "cpp":          ["decode_tensor_viterbi_cpp"],
    "omp":          ["decode_tensor_viterbi_omp"],
    "gpu":          ["decode_tensor_viterbi_cuda"],
    "baseline":     ["HSMMLearn_CPP", "HSMMLearn_OMP"],
    "baseline-cpp": ["HSMMLearn_CPP"],
    "baseline-omp": ["HSMMLearn_OMP"],
}

# flag name -> walltimes.yaml version key(s). "baseline" expands to both
# baseline backends, same as _FNAME_FOR_FLAG above.
_VERSIONS_FOR_FLAG = {
    "py":           ["py"],
    "cpp":          ["cpp"],
    "omp":          ["omp"],
    "gpu":          ["gpu"],
    "baseline":     ["baseline-cpp", "baseline-omp"],
    "baseline-cpp": ["baseline-cpp"],
    "baseline-omp": ["baseline-omp"],
}


def expand_versions(viterbi_flags: str) -> list[str]:
    out: list[str] = []
    for f in viterbi_flags.split(":"):
        out.extend(_VERSIONS_FOR_FLAG.get(f, []))
    return out


def compute_viterbi_flags(args, sys_type: str) -> str:
    flag_map = [
        ("py",           getattr(args, "py", False)),
        ("cpp",          getattr(args, "cpp", False)),
        ("omp",          getattr(args, "omp", False)),
        ("gpu",          getattr(args, "gpu", False)),
        ("baseline",     getattr(args, "baseline", False)),
        ("baseline-cpp", getattr(args, "baseline_cpp", False)),
        ("baseline-omp", getattr(args, "baseline_omp", False)),
    ]
    viterbi_flags = ":".join(name for name, enabled in flag_map if enabled)
    if not viterbi_flags:
        viterbi_flags = "gpu" if sys_type == "gpu" else "cpp:omp:baseline-cpp:baseline-omp"
    return viterbi_flags


def expected_csv_stems(flags: str) -> list[str]:
    out: list[str] = []
    for f in flags.split(":"):
        out.extend(_FNAME_FOR_FLAG.get(f, []))
    return out


def selected_likwid_cpu_flags(args) -> list[str] | None:
    cpu_flag_map = [
        ("--baseline",     getattr(args, "baseline", False)),
        ("--baseline-omp", getattr(args, "baseline_omp", False)),
        ("--cpp",          getattr(args, "cpp", False)),
        ("--omp",          getattr(args, "omp", False)),
    ]
    selected = [f for f, enabled in cpu_flag_map if enabled]
    return selected or None
