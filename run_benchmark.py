#!/usr/bin/env python3
"""
run_benchmark.py — SLURM job submitter for tensor-viterbi benchmarks.

Generates a temporary .slrm script and submits a grid of jobs (or runs locally).

Usage:
  ./run_benchmark.py --system <sys> --toolchain <tc> [backend flags]

Examples:
  ./run_benchmark.py --system xeon8480 --toolchain intel --cpp --omp --baseline
  ./run_benchmark.py --system a100 --toolchain cuda
  ./run_benchmark.py --system epyc-9474f --toolchain amd
  ./run_benchmark.py --system epyc-7763-bigmem --toolchain all
"""

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from compile import compile_system

SCRIPT_DIR = Path(__file__).resolve().parent

def _load_benchmark_params() -> tuple[list[int], list[int], list[int]]:
    cfg = SCRIPT_DIR / "benchmark_params.cfg"
    params: dict[str, list[int]] = {}
    for line in cfg.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.partition("=")
        params[key.strip()] = [int(v) for v in raw.split(",") if v.strip()]
    try:
        return params["states"], params["durations"], params["timesteps"]
    except KeyError as e:
        print(f"Error: benchmark_params.cfg is missing key {e}")
        sys.exit(1)


STATES, DURATIONS, TIMESTEPS = _load_benchmark_params()

TMP_SLRM           = SCRIPT_DIR / ".tmp_benchmark.slrm"
TMP_LIKWID_SLRM    = SCRIPT_DIR / ".tmp_likwid.slrm"
LIKWID_DATA        = "data/75states_1000steps_500dur.json"
LIKWID_PERF_GROUPS = ["FLOPS_DP", "MEM", "L3", "L2", "BRANCH"]
LIKWID_CPU_FLAGS   = ["--baseline", "--baseline-omp", "--cpp", "--omp"]

# ── Requirements check ───────────────────────────────────────────────────────

def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[.+]", v) if x.isdigit())


def check_requirements() -> None:
    ops = {
        ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,  "<":  lambda a, b: a < b,
    }
    req_file = SCRIPT_DIR / "requirements.txt"
    missing: list[str] = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*([><=!]{1,2})\s*([0-9][0-9.]*)", line)
        pkg, op, req_ver = (m.group(1) if m else line), (m.group(2) if m else None), (m.group(3) if m else None)
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"  {pkg}: not installed")
            continue
        if op and req_ver and not ops[op](_ver(installed), _ver(req_ver)):
            missing.append(f"  {pkg}: need {op}{req_ver}, got {installed}")
    if missing:
        print("Error: missing or outdated requirements:")
        print("\n".join(missing))
        print(f"Install them with: pip install -r {req_file}")
        sys.exit(1)


def get_available_likwid_groups() -> set[str]:
    """Query likwid-perfctr -a for available performance groups on this CPU."""
    try:
        result = subprocess.run(
            ["likwid-perfctr", "-a"], capture_output=True, text=True
        )
        groups = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and re.match(r"^[A-Z0-9_]+$", parts[0]):
                groups.add(parts[0])
        return groups
    except FileNotFoundError:
        print("[✗] likwid-perfctr not found in PATH", file=sys.stderr)
        sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────

def load_systems() -> dict:
    with open(SCRIPT_DIR / "systems.json") as f:
        return json.load(f)


# ── Walltime ─────────────────────────────────────────────────────────────────

def get_walltime(s: int, d: int, t: int) -> str:
    if t <= 10_000:
        if s == 75 and d == 1000 and t == 10_000:
            return "02:00:00"
        return "00:30:00"
    if t == 100_000:
        if s <= 15:
            return "01:00:00"
        if s == 25:
            return "01:00:00" if d <= 250 else "02:00:00"
        if s == 50:
            return {100: "01:00:00", 250: "02:00:00", 500: "04:00:00"}.get(d, "08:00:00")
        if s == 75:
            return {100: "02:00:00", 250: "04:00:00", 500: "08:00:00"}.get(d, "16:00:00")
        return "01:00:00"
    if t == 1_000_000:
        if s <= 15:
            return "02:00:00"
        if s == 25:
            return "02:00:00" if d <= 250 else "06:00:00"
        if s == 50:
            return {100: "02:00:00", 250: "08:00:00", 500: "14:00:00"}.get(d, "20:00:00")
        if s == 75:
            return {100: "05:00:00", 250: "10:00:00", 500: "20:00:00"}.get(d, "20:00:00")
        return "02:00:00"
    if t == 10_000_000:
        return "20:00:00"
    return "20:00:00"


# ── SLURM script generation ───────────────────────────────────────────────────

def generate_slrm() -> Path:
    """Write the temporary SLURM job script that calls viterbi_app.py."""
    TMP_SLRM.write_text("""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
# --job-name, --output, --error, --partition, --account, --time,
# --cpus-per-task / --gres are injected by run_benchmark.py via sbatch CLI.
# SYS_NAME, SYS_TYPE, SYS_MODULES, SYS_UENV, SYS_OMP_BIND, SYS_OMP_PLACES,
# SYS_CPUS, SYS_METRICS_BACKEND, VITERBI_FLAGS, BENCHMARK_ITERATIONS, DATA_PATH
# are passed via --export.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -n "${SYS_UENV:-}" && -z "${_UENV_ACTIVE:-}" ]]; then
    exec uenv run --view=modules "$SYS_UENV" -- env _UENV_ACTIVE=1 bash "$0" "$@"
fi

if [[ "$SYS_TYPE" == "cpu" ]]; then
    lscpu | grep -E "Model name|CPU\\(s\\):|Thread\\(s\\) per core|Core\\(s\\) per socket"
    grep MemTotal /proc/meminfo
elif [[ "$SYS_TYPE" == "gpu" ]]; then
    command -v nvidia-smi &>/dev/null && \\
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    command -v rocm-smi &>/dev/null && \\
        rocm-smi --showproductname --showmeminfo vram --showdriverversion
fi

IFS=':' read -ra _MODS <<< "$SYS_MODULES"
for _mod in "${_MODS[@]}"; do [[ -n "$_mod" ]] && module load "$_mod"; done

if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
elif [[ -n "$SYS_CPUS" ]]; then
    export OMP_NUM_THREADS=$SYS_CPUS
fi
[[ -n "${SYS_OMP_BIND:-}" ]]   && export OMP_PROC_BIND="$SYS_OMP_BIND"
[[ -n "${SYS_OMP_PLACES:-}" ]] && export OMP_PLACES="$SYS_OMP_PLACES"

cd "$SCRIPT_DIR"

IFS='/' read -r _SYS _TC <<< "$SYS_NAME"
PYTHON_FLAGS=""
IFS=':' read -ra _VFLAGS <<< "$VITERBI_FLAGS"
for _f in "${_VFLAGS[@]}"; do [[ -n "$_f" ]] && PYTHON_FLAGS="$PYTHON_FLAGS --$_f"; done

python viterbi_app.py $PYTHON_FLAGS \\
    --system "$_SYS" --toolchain "$_TC" \\
    --iterations "${BENCHMARK_ITERATIONS:-6}" \\
    --data-path "$DATA_PATH"
""")
    TMP_SLRM.chmod(0o755)
    return TMP_SLRM


def generate_likwid_slrm() -> Path:
    """Write the SLURM script for LIKWID hardware-counter profiling."""
    groups_bash = " ".join(LIKWID_PERF_GROUPS)
    TMP_LIKWID_SLRM.write_text("""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
# SYS_NAME, SYS_MODULES, SYS_UENV, SYS_OMP_BIND, SYS_OMP_PLACES, SYS_CPUS
# are passed via --export.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -n "${SYS_UENV:-}" && -z "${_UENV_ACTIVE:-}" ]]; then
    exec uenv run --view=modules "$SYS_UENV" -- env _UENV_ACTIVE=1 bash "$0" "$@"
fi

lscpu | grep -E "Model name|CPU\\(s\\):|Thread\\(s\\) per core|Core\\(s\\) per socket"
grep MemTotal /proc/meminfo

IFS=':' read -ra _MODS <<< "$SYS_MODULES"
for _mod in "${_MODS[@]}"; do [[ -n "$_mod" ]] && module load "$_mod"; done

if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
elif [[ -n "$SYS_CPUS" ]]; then
    export OMP_NUM_THREADS=$SYS_CPUS
fi
[[ -n "${SYS_OMP_BIND:-}" ]]   && export OMP_PROC_BIND="$SYS_OMP_BIND"
[[ -n "${SYS_OMP_PLACES:-}" ]] && export OMP_PLACES="$SYS_OMP_PLACES"

cd "$SCRIPT_DIR"

IFS='/' read -r _SYS _TC <<< "$SYS_NAME"
OUTDIR="${SCRIPT_DIR}/results/${SYS_NAME}"
mkdir -p "$OUTDIR"
DATA="${SCRIPT_DIR}/data/75states_1000steps_500dur.json"

declare -a PERF_GROUPS=()
for _g in {groups_bash}; do
    if likwid-perfctr -a 2>/dev/null | grep -qE "^${{_g}}[[:space:]]"; then
        PERF_GROUPS+=("$_g")
    else
        echo "[!] Skipping group $_g: not available on this CPU" >&2
    fi
done
                               
for VERSION_FLAG in --baseline --baseline-omp --cpp --omp; do
    VERSION_NAME="${VERSION_FLAG#--}"
    OUTPUT_FILE="${OUTDIR}/likwid_${VERSION_NAME}.txt"
    > "$OUTPUT_FILE"

    for GROUP in "${PERF_GROUPS[@]}"; do
        echo "-> ${VERSION_NAME} / ${GROUP}"
        LIKWID_CSV="${OUTDIR}/likwid_${VERSION_NAME}_${GROUP}.csv"
                               
        likwid-perfctr -C 0 -g "${GROUP}" -m \\
            -o "${LIKWID_CSV}" \\
            -- python "${SCRIPT_DIR}/viterbi_app.py" \\
               --system "${_SYS}_likwid" --toolchain "${_TC}" \\
               --iterations 1 \\
               "${VERSION_FLAG}" \\
               --data-path "${DATA}" \\
            >> "$OUTPUT_FILE" \\
            2>>"${OUTDIR}/likwid.log"
    done
done
""")
    TMP_LIKWID_SLRM.chmod(0o755)
    return TMP_LIKWID_SLRM


# ── Job environment ───────────────────────────────────────────────────────────

def _build_job_env(sys_info: dict, vflags: str, iters: int, data_path: str) -> dict:
    return {
        "SYS_NAME":             sys_info["sys_name"],
        "SYS_TYPE":             sys_info["type"],
        "SYS_MODULES":          sys_info.get("modules", ""),
        "SYS_METRICS_BACKEND":  sys_info.get("metrics_backend", ""),
        "SYS_UENV":             sys_info.get("uenv", ""),
        "SYS_OMP_BIND":         sys_info.get("omp_bind", ""),
        "SYS_OMP_PLACES":       sys_info.get("omp_places", ""),
        "SYS_CPUS":             str(sys_info.get("cpus", "")),
        "VITERBI_FLAGS":        vflags,
        "BENCHMARK_ITERATIONS": str(iters),
        "DATA_PATH":            data_path,
    }


# ── Submission helpers ────────────────────────────────────────────────────────


def submit_slurm(
    stem: str, vflags: str, iters: int, walltime: str, data_path: str,
    sbatch_flags: list, sys_info: dict, results_dir: Path,
) -> None:
    job_env    = _build_job_env(sys_info, vflags, iters, data_path)
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in job_env.items())
    cmd = [
        "sbatch", *sbatch_flags,
        f"--export={export_str}",
        f"--job-name=tv_{stem}",
        f"--time={walltime}",
        f"--output={results_dir / (stem + '.out')}",
        f"--error={results_dir / (stem + '.err')}",
        str(TMP_SLRM),
    ]
    print(f"  -> flags=[{vflags}] iterations={iters}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    print(output)
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return
    time.sleep(0.1)


def run_local(
    stem: str, vflags: str, iters: int, data_path: str,
    sys_info: dict, results_dir: Path,
) -> None:
    system, toolchain = sys_info["sys_name"].split("/", 1)
    backend_flags = [f"--{f}" for f in vflags.split(":") if f]
    cmd = [
        "python", str(SCRIPT_DIR / "viterbi_app.py"),
        "--system", system, "--toolchain", toolchain,
        "--iterations", str(iters),
        "--data-path", data_path,
        *backend_flags,
    ]
    cpus = sys_info.get("cpus")
    env = {**os.environ, "SYS_METRICS_BACKEND": sys_info.get("metrics_backend", "")}
    if cpus:
        env["OMP_NUM_THREADS"] = str(cpus)
    if sys_info.get("omp_bind"):
        env["OMP_PROC_BIND"] = sys_info["omp_bind"]
    if sys_info.get("omp_places"):
        env["OMP_PLACES"] = sys_info["omp_places"]
    print(f"  -> flags=[{vflags}] iterations={iters} (local)")
    with open(results_dir / f"{stem}.out", "w") as fout, \
         open(results_dir / f"{stem}.err", "w") as ferr:
        subprocess.run(cmd, env=env, stdout=fout, stderr=ferr, cwd=str(SCRIPT_DIR))


def run_likwid_local(sys_info: dict, results_dir: Path) -> None:
    _system, toolchain = sys_info["sys_name"].split("/", 1)
    system = f"{_system}_likwid"
    data     = str(SCRIPT_DIR / LIKWID_DATA)
    log_file = results_dir / "likwid.log"

    env = {**os.environ}
    if sys_info.get("cpus"):
        env["OMP_NUM_THREADS"] = str(sys_info["cpus"])
    if sys_info.get("omp_bind"):
        env["OMP_PROC_BIND"] = sys_info["omp_bind"]
    if sys_info.get("omp_places"):
        env["OMP_PLACES"] = sys_info["omp_places"]

    available = get_available_likwid_groups()
    groups_to_run = []
    for g in LIKWID_PERF_GROUPS:
        if g in available:
            groups_to_run.append(g)
        else:
            print(f"  [!] Skipping group {g}: not available on this CPU")

    if not groups_to_run:
        print("[✗] No requested LIKWID groups are available on this CPU.", file=sys.stderr)
        return

    for version_flag in LIKWID_CPU_FLAGS:
        version_name = version_flag.lstrip("-")
        output_file  = results_dir / f"likwid_{version_name}.txt"
        output_file.write_text("")

        for group in groups_to_run:
            print(f"  -> {version_name} / {group}")
            likwid_csv = log_file.parent / f"likwid_{version_name}_{group}.csv"

            cmd = [
                "likwid-perfctr", "-C", "0", "-g", group, "-m",
                "-o", str(likwid_csv),
                "--",
                "python", str(SCRIPT_DIR / "viterbi_app.py"),
                "--system", system, "--toolchain", toolchain,
                "--iterations", "1",
                version_flag,
                "--data-path", data,
            ]

            result = subprocess.run(
                cmd, env=env, cwd=str(SCRIPT_DIR), capture_output=True, text=True)

            with open(log_file, "a") as flog:
                flog.write(result.stderr)

            with open(output_file, "a") as fout:
                fout.write(result.stdout)


def submit_likwid_slurm(sys_info: dict, sbatch_flags: list, results_dir: Path) -> None:
    job_env = {
        "SYS_NAME":       sys_info["sys_name"],
        "SYS_TYPE":       sys_info["type"],
        "SYS_MODULES":    sys_info.get("modules", ""),
        "SYS_UENV":       sys_info.get("uenv", ""),
        "SYS_OMP_BIND":   sys_info.get("omp_bind", ""),
        "SYS_OMP_PLACES": sys_info.get("omp_places", ""),
        "SYS_CPUS":       str(sys_info.get("cpus", "")),
    }
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in job_env.items())
    stem = "likwid"
    cmd = [
        "sbatch", *sbatch_flags,
        f"--export={export_str}",
        f"--job-name=tv_{stem}",
        "--time=02:00:00",
        f"--output={results_dir / (stem + '.out')}",
        f"--error={results_dir / (stem + '.err')}",
        str(TMP_LIKWID_SLRM),
    ]
    print(f"  -> Submitting LIKWID profiling job for {sys_info['sys_name']}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(system: str, toolchain: str, sys_conf: dict, tc_conf: dict, args) -> None:
    sys_name  = f"{system}/{toolchain}"
    sys_type  = sys_conf["type"]
    scheduler = sys_conf.get("scheduler", "slurm")

    sys_info = {
        "sys_name":        sys_name,
        "type":            sys_type,
        "modules":         tc_conf.get("modules", ""),
        "metrics_backend": tc_conf.get("metrics_backend", ""),
        "uenv":            tc_conf.get("uenv", ""),
        "omp_bind":        sys_conf.get("omp_bind", ""),
        "omp_places":      sys_conf.get("omp_places", ""),
        "cpus":            sys_conf.get("cpus", ""),
    }

    flag_map = [
        ("py",           args.py),
        ("cpp",          args.cpp),
        ("omp",          args.omp),
        ("cuda",         args.cuda),
        ("baseline",     args.baseline),
        ("baseline-cpp", args.baseline_cpp),
        ("baseline-omp", args.baseline_omp),
    ]
    viterbi_flags = ":".join(name for name, enabled in flag_map if enabled)
    if not viterbi_flags:
        if sys_type == "gpu":
            viterbi_flags = "cuda"
        else:
            viterbi_flags = "cpp:omp:baseline-cpp:baseline-omp"

    results_dir = SCRIPT_DIR / "results" / sys_name
    results_dir.mkdir(parents=True, exist_ok=True)

    sbatch_flags: list[str] = []
    if scheduler == "slurm":
        if "account" in sys_conf:
            sbatch_flags.append(f"--account={sys_conf['account']}")
        if sys_conf.get("partition"):
            sbatch_flags.append(f"--partition={sys_conf['partition']}")
        if "qos" in sys_conf:
            sbatch_flags.append(f"--qos={sys_conf['qos']}")
        sbatch_flags.append("--gres=gpu:1" if sys_type == "gpu"
                            else f"--cpus-per-task={sys_conf.get('cpus', 1)}")

    def submit(job_stem: str, flags: str, job_iters: int, walltime: str, config_file: str) -> None:
        if scheduler == "local":
            run_local(job_stem, flags, job_iters, config_file, sys_info, results_dir)
        else:
            submit_slurm(job_stem, flags, job_iters, walltime, config_file,
                         sbatch_flags, sys_info, results_dir)

    for s in STATES:
        for d in DURATIONS:
            for t in TIMESTEPS:
                walltime    = get_walltime(s, d, t)
                stem        = f"{s}s_{d}d_{t}t"
                config_file = str(SCRIPT_DIR / f"data/{s}states_{t}steps_{d}dur.json")
                print(f"Submitting: System={system}, States={s}, Duration={d}, "
                      f"Timesteps={t}, Walltime={walltime}")

                iters = args.iterations
                if t >= 1_000_000 and iters > 2:
                    iters = 2

  
                if scheduler == "local":
                    run_local(stem, viterbi_flags, iters, config_file, sys_info, results_dir)
                else:
                    submit_slurm(stem, viterbi_flags, iters, walltime, config_file,
                                sbatch_flags, sys_info, results_dir)


def run_likwid_profiling(system: str, toolchain: str, sys_conf: dict, tc_conf: dict) -> None:
    sys_name  = f"{system}/{toolchain}"
    sys_type  = sys_conf["type"]
    scheduler = sys_conf.get("scheduler", "slurm")

    if sys_type != "cpu":
        print(f"Warning: LIKWID profiling is CPU-only (system type={sys_type}). Skipping {sys_name}.")
        return

    sys_info = {
        "sys_name":    sys_name,
        "type":        sys_type,
        "modules":     tc_conf.get("modules", ""),
        "uenv":        tc_conf.get("uenv", ""),
        "omp_bind":    sys_conf.get("omp_bind", ""),
        "omp_places":  sys_conf.get("omp_places", ""),
        "cpus":        sys_conf.get("cpus", ""),
    }

    results_dir = SCRIPT_DIR / "results" / sys_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"LIKWID profiling: {sys_name}")

    if scheduler == "local":
        run_likwid_local(sys_info, results_dir)
    else:
        sbatch_flags: list[str] = []
        if "account" in sys_conf:
            sbatch_flags.append(f"--account={sys_conf['account']}")
        if sys_conf.get("partition"):
            sbatch_flags.append(f"--partition={sys_conf['partition']}")
        if "qos" in sys_conf:
            sbatch_flags.append(f"--qos={sys_conf['qos']}")
        sbatch_flags.append(f"--cpus-per-task={sys_conf.get('cpus', 1)}")
        submit_likwid_slurm(sys_info, sbatch_flags, results_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="tensor-viterbi SLURM job submitter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system",       required=True, help="System key from systems.json")
    parser.add_argument("--toolchain",    required=True, help="Toolchain key, or 'all'")
    parser.add_argument("--py",           action="store_true")
    parser.add_argument("--cpp",          action="store_true")
    parser.add_argument("--omp",          action="store_true")
    parser.add_argument("--cuda",         action="store_true")
    parser.add_argument("--baseline",     action="store_true")
    parser.add_argument("--baseline-cpp", action="store_true", dest="baseline_cpp")
    parser.add_argument("--baseline-omp", action="store_true", dest="baseline_omp")
    parser.add_argument("--iterations",   type=int, default=6, metavar="N")
    parser.add_argument("--likwid", action="store_true",
                        help="Run LIKWID hardware-counter profiling (CPU only, fixed data file, 1 iteration)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        print("Error: must be run from the repository root.")
        print(f"  cd {SCRIPT_DIR} && python {Path(__file__).name} {' '.join(sys.argv[1:])}")
        sys.exit(1)

    check_requirements()

    systems = load_systems()
    if args.system not in systems:
        print(f"Error: Unknown system '{args.system}'.")
        print(f"Available systems: {', '.join(systems)}")
        sys.exit(1)

    sys_conf   = systems[args.system]
    toolchains = sys_conf.get("toolchains", {})

    if sys_conf.get("scheduler", "slurm") == "slurm":
        generate_slrm()
        if args.likwid:
            generate_likwid_slrm()

    if args.toolchain == "all":
        if not toolchains:
            print(f"Error: No toolchains defined for system '{args.system}'.")
            sys.exit(1)
        for tc in sorted(toolchains):
            print(f"=== Compiling {args.system} / {tc} ===")
            compile_system(args.system, tc, sys_conf, toolchains[tc], args.likwid)
            print(f"=== Submitting {args.system} / {tc} ===")
            if args.likwid:
                run_likwid_profiling(args.system, tc, sys_conf, toolchains[tc])
            else:
                run_sweep(args.system, tc, sys_conf, toolchains[tc], args)
        return

    if args.toolchain not in toolchains:
        print(f"Error: Toolchain '{args.toolchain}' not defined for system '{args.system}'.")
        print(f"Known toolchains: {', '.join(toolchains)}")
        sys.exit(1)

    print(f"=== Compiling {args.system} / {args.toolchain} ===")
    compile_system(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args.likwid)
    if args.likwid:
        run_likwid_profiling(args.system, args.toolchain, sys_conf, toolchains[args.toolchain])
    else:
        run_sweep(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args)


if __name__ == "__main__":
    main()
