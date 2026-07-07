#!/usr/bin/env python3
"""
run_benchmark.py — sole entry point for tensor-viterbi: compiles the native
extension (via compile.py) then dispatches benchmark jobs (sbatch for SLURM
systems, direct call for local systems).

--pack is required: it selects which walltime bucket of the
(states, duration, timesteps) grid to run (see PACKS below).

Usage:
  ./run_benchmark.py --system <sys> --toolchain <tc> --scheduler <local|slurm> --pack <pack> [backend flags]

Examples:
  ./run_benchmark.py --system xeon8480 --toolchain intel --scheduler slurm --pack 1h --cpp --omp --baseline
  ./run_benchmark.py --system a100 --toolchain cuda --scheduler slurm --pack 2h
  ./run_benchmark.py --system epyc-9474f --toolchain amd --scheduler local --pack 4-8h
  ./run_benchmark.py --system epyc-7763-bigmem --toolchain all --scheduler slurm --pack 10-20h
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

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
LIKWID_DATA        = "data/75states_10000steps_1000dur.json"
LIKWID_PERF_GROUPS = ["FLOPS_DP", "MEM", "L3", "L2", "BRANCH", "TMA"]
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
        if m:
            pkg, op, req_ver = m.group(1), m.group(2), m.group(3)
        else:
            pkg, op, req_ver = line, None, None
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
    """Merge architectures.yaml (machine descriptors) with slurm_system.yaml
    (SLURM partition/account/qos and per-toolchain modules/uenv) into one
    system-config dict, keyed and shaped exactly as the rest of the script expects."""
    with open(SCRIPT_DIR / "architectures.yaml") as f:
        architectures = yaml.safe_load(f) or {}
    with open(SCRIPT_DIR / "slurm_system.yaml") as f:
        slurm = yaml.safe_load(f) or {}

    systems: dict = {}
    for name, arch_conf in architectures.items():
        slurm_conf = slurm.get(name, {})
        conf = {**arch_conf, **{k: v for k, v in slurm_conf.items() if k != "toolchains"}}

        arch_toolchains  = arch_conf.get("toolchains", {}) or {}
        slurm_toolchains = slurm_conf.get("toolchains", {}) or {}
        conf["toolchains"] = {
            tc: {**(arch_toolchains.get(tc) or {}), **(slurm_toolchains.get(tc) or {})}
            for tc in {**arch_toolchains, **slurm_toolchains}
        }
        systems[name] = conf
    return systems


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
    return "20:00:00"


# ── Walltime packs ───────────────────────────────────────────────────────────
# Buckets the (states, duration, timesteps) grid by get_walltime() so an
# evaluator can pick how much wall-clock budget to spend. Boundaries follow
# the actual value set get_walltime() produces (30m/1h/2h/4h/5h/6h/8h/10h/
# 14h/16h/20h) — there's a natural gap between 8h and 10h, so "4-8h" and
# "10-20h" don't overlap or leave holes.
PACKS: dict[str, tuple[int, int]] = {
    "1h":     (0,             1 * 3600),
    "2h":     (1 * 3600 + 1,  2 * 3600),
    "4-8h":   (2 * 3600 + 1,  8 * 3600),
    "10-20h": (8 * 3600 + 1, 20 * 3600),
}


def _hms_to_seconds(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def _pack_of_walltime(walltime: str) -> str | None:
    seconds = _hms_to_seconds(walltime)
    for name, (lo, hi) in PACKS.items():
        if lo <= seconds <= hi:
            return name
    return None


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

RUNNER=""
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
    RUNNER="nsys profile -f true -o ${RESULTS_DIR}/${SLURM_JOB_NAME}"
fi
if [[ "${NCU_PROFILE:-0}" == "1" ]]; then
    RUNNER="ncu --target-processes all --set full --force-overwrite --launch-skip 1500 --launch-count 10 -o ${RESULTS_DIR}/${SLURM_JOB_NAME}"
fi

$RUNNER python viterbi_app.py $PYTHON_FLAGS \\
    --system "$_SYS" --toolchain "$_TC" \\
    --iterations "${BENCHMARK_ITERATIONS:-6}" \\
    --data-path "$DATA_PATH"
""")
    TMP_SLRM.chmod(0o755)
    return TMP_SLRM


def generate_likwid_slrm(cpu_flags: list[str] | None = None) -> Path:
    """Write the SLURM script for LIKWID hardware-counter profiling."""
    groups_bash = " ".join(LIKWID_PERF_GROUPS)
    if not cpu_flags:
        cpu_flags = LIKWID_CPU_FLAGS
    
    script_template = """\
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
for _g in TARGET_GROUPS_PLACEHOLDER; do
    if likwid-perfctr -a 2>/dev/null | grep -qE "^[[:space:]]*${_g}[[:space:]]"; then
        PERF_GROUPS+=("$_g")
    else
        echo "[!] Skipping group $_g: not available on this CPU" >&2
    fi
done               

for VERSION_FLAG in VERSION_FLAGS_PLACEHOLDER; do
    VERSION_NAME="${VERSION_FLAG#--}"
    OUTPUT_FILE="${OUTDIR}/likwid_${VERSION_NAME}.txt"
    > "$OUTPUT_FILE"

    for GROUP in "${PERF_GROUPS[@]}"; do
        echo "-> ${VERSION_NAME} / ${GROUP}"
        LIKWID_CORES="E:N:${OMP_NUM_THREADS}"
        LIKWID_CSV="${OUTDIR}/likwid_${VERSION_NAME}_${GROUP}.csv"
                               
        likwid-perfctr -C "${LIKWID_CORES}" -g "${GROUP}" -m \\
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
"""
    
    flags_bash = " ".join(cpu_flags)
    final_script = (
        script_template
        .replace("TARGET_GROUPS_PLACEHOLDER", groups_bash)
        .replace("VERSION_FLAGS_PLACEHOLDER", flags_bash)
    )
    
    TMP_LIKWID_SLRM.write_text(final_script)
    TMP_LIKWID_SLRM.chmod(0o755)
    return TMP_LIKWID_SLRM


# ── Job environment / SLURM helpers ──────────────────────────────────────────

def _build_sys_info(system: str, toolchain: str, sys_conf: dict, tc_conf: dict) -> dict:
    return {
        "sys_name":        f"{system}/{toolchain}",
        "type":            sys_conf["type"],
        "modules":         tc_conf.get("modules", ""),
        "metrics_backend": tc_conf.get("metrics_backend", ""),
        "uenv":            tc_conf.get("uenv", ""),
        "omp_bind":        sys_conf.get("omp_bind", ""),
        "omp_places":      sys_conf.get("omp_places", ""),
        "cpus":            sys_conf.get("cpus", ""),
    }


def _build_sbatch_flags(sys_conf: dict, sys_type: str) -> list[str]:
    flags: list[str] = []
    if "account" in sys_conf:
        flags.append(f"--account={sys_conf['account']}")
    if sys_conf.get("partition"):
        flags.append(f"--partition={sys_conf['partition']}")
    if "qos" in sys_conf:
        flags.append(f"--qos={sys_conf['qos']}")
    flags.append("--gres=gpu:1" if sys_type == "gpu" else f"--cpus-per-task={sys_conf.get('cpus', 1)}")
    return flags


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
    nsys: bool = False, ncu: bool = False,
) -> None:
    job_env    = _build_job_env(sys_info, vflags, iters, data_path)
    job_env["NSYS_PROFILE"] = "1" if nsys else "0"
    job_env["NCU_PROFILE"]  = "1" if ncu  else "0"
    job_env["RESULTS_DIR"]  = str(results_dir)
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
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return
    time.sleep(0.1)


def run_local(
    stem: str, vflags: str, iters: int, data_path: str,
    sys_info: dict, results_dir: Path, nsys: bool = False, ncu: bool = False,
) -> None:
    system, toolchain = sys_info["sys_name"].split("/", 1)
    backend_flags = [f"--{f}" for f in vflags.split(":") if f]
    if ncu:
        prefix = ["ncu", "--target-processes", "all", "--set", "full", "--force-overwrite", "--launch-skip", "1500", "--launch-count", "10", "-o", str(results_dir / stem)]
    elif nsys:
        prefix = ["nsys", "profile", "-f", "true", "-o", str(results_dir / stem)]
    else:
        prefix = []
    cmd = [
        *prefix, "python", str(SCRIPT_DIR / "viterbi_app.py"),
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


def run_likwid_local(sys_info: dict, results_dir: Path, cpu_flags: list[str] | None = None) -> None:
    base_system, toolchain = sys_info["sys_name"].split("/", 1)
    system = f"{base_system}_likwid"
    data   = str(SCRIPT_DIR / LIKWID_DATA)
    log_file = results_dir / "likwid.log"

    env = {**os.environ}
    num_threads = sys_info.get("cpus") or 1
    env["OMP_NUM_THREADS"] = str(num_threads)
    if sys_info.get("omp_bind"):
        env["OMP_PROC_BIND"] = sys_info["omp_bind"]
    if sys_info.get("omp_places"):
        env["OMP_PLACES"] = sys_info["omp_places"]
    likwid_cores = f"E:N:{num_threads}"

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

    for version_flag in (cpu_flags or LIKWID_CPU_FLAGS):
        version_name = version_flag.lstrip("-")
        output_file  = results_dir / f"likwid_{version_name}.txt"
        output_file.write_text("")

        for group in groups_to_run:
            print(f"  -> {version_name} / {group}")
            likwid_csv = results_dir / f"likwid_{version_name}_{group}.csv"

            cmd = [
                "likwid-perfctr", "-C", likwid_cores, "-g", group, "-m",
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
    sys_info  = _build_sys_info(system, toolchain, sys_conf, tc_conf)
    sys_name  = sys_info["sys_name"]
    sys_type  = sys_info["type"]
    scheduler = args.scheduler

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

    pack = args.pack
    print(f"Pack selected: {pack} "
          f"({PACKS[pack][0] // 3600}h–{PACKS[pack][1] // 3600}h walltime range)")

    sbatch_flags = _build_sbatch_flags(sys_conf, sys_type) if scheduler == "slurm" else []

    def submit(job_stem: str, flags: str, job_iters: int, walltime: str, config_file: str) -> None:
        if scheduler == "local":
            run_local(job_stem, flags, job_iters, config_file, sys_info, results_dir,
                      nsys=args.nsys, ncu=args.ncu)
        else:
            submit_slurm(job_stem, flags, job_iters, walltime, config_file,
                         sbatch_flags, sys_info, results_dir, nsys=args.nsys, ncu=args.ncu)

    skipped = 0

    for s in STATES:
        for d in DURATIONS:
            for t in TIMESTEPS:
                walltime = get_walltime(s, d, t)
                if _pack_of_walltime(walltime) != pack:
                    skipped += 1
                    continue
                stem        = f"{s}s_{d}d_{t}t"
                config_file = str(SCRIPT_DIR / f"data/{s}states_{t}steps_{d}dur.json")
                print(f"Submitting: System={system}, States={s}, Duration={d}, "
                      f"Timesteps={t}, Walltime={walltime}")

                iters = args.iterations
                if t >= 1_000_000 and iters > 2:
                    iters = 2

                submit(stem, viterbi_flags, iters, walltime, config_file)

    print(f"Pack '{pack}': skipped {skipped} job(s) outside the selected walltime range.")


def run_likwid_profiling(system: str, toolchain: str, sys_conf: dict, tc_conf: dict,
                         scheduler: str, cpu_flags: list[str] | None = None) -> None:
    if sys_conf["type"] != "cpu":
        print(f"Warning: LIKWID profiling is CPU-only (system type={sys_conf['type']}). "
              f"Skipping {system}/{toolchain}.")
        return

    sys_info  = _build_sys_info(system, toolchain, sys_conf, tc_conf)

    results_dir = SCRIPT_DIR / "results" / sys_info["sys_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"LIKWID profiling: {sys_info['sys_name']}")

    if scheduler == "local":
        run_likwid_local(sys_info, results_dir, cpu_flags)
    else:
        submit_likwid_slurm(sys_info, _build_sbatch_flags(sys_conf, "cpu"), results_dir)





# ---------------------------
# ENTRY POINT
# ---------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="tensor-viterbi SLURM job submitter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system",    "-s", required=True,
                        help="System key from architectures.yaml")
    parser.add_argument("--toolchain", "-t", required=True,
                        help="Toolchain key, or 'all' to build every toolchain for the system")
    parser.add_argument("--scheduler", choices=["local", "slurm"], required=True,
                        help="'local' calls viterbi_app.py directly; 'slurm' generates a "
                             ".slrm script and submits it via sbatch")
    parser.add_argument("--py",           action="store_true")
    parser.add_argument("--cpp",          action="store_true")
    parser.add_argument("--omp",          action="store_true")
    parser.add_argument("--cuda",         action="store_true")
    parser.add_argument("--baseline",     action="store_true")
    parser.add_argument("--baseline-cpp", action="store_true", dest="baseline_cpp")
    parser.add_argument("--baseline-omp", action="store_true", dest="baseline_omp")
    parser.add_argument("--iterations",   type=int, default=6, metavar="N")
    parser.add_argument("--nsys", action="store_true",
                        help="Wrap runs with nsys profile (CUDA timeline tracing)")
    parser.add_argument("--ncu", action="store_true",
                        help="Wrap runs with ncu (Nsight Compute kernel profiling; overrides --nsys)")
    parser.add_argument("--pack", choices=list(PACKS), default=None,
                        help="Required unless --likwid is given. Only submit jobs whose "
                             f"estimated walltime falls in this pack ({', '.join(PACKS)}). "
                             "Ignored for --likwid runs.")
    parser.add_argument("--likwid", action="store_true",
                        help="Run LIKWID hardware-counter profiling (CPU only, fixed data file, 1 iteration)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.likwid and args.pack is None:
        parser.error("--pack is required unless --likwid is given")

    if sys.version_info < (3, 10):
        print(f"Error: Python >= 3.10 required, found {sys.version.split()[0]}.")
        print("Activate a Python 3.10+ environment before running run_benchmark.py.")
        sys.exit(1)

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

    cpu_flag_map = [
        ("--baseline",     args.baseline),
        ("--baseline-omp", args.baseline_omp),
        ("--cpp",          args.cpp),
        ("--omp",          args.omp),
    ]
    selected_cpu_flags = [f for f, enabled in cpu_flag_map if enabled] or None

    if args.scheduler == "slurm":
        generate_slrm()
        if args.likwid:
            generate_likwid_slrm(selected_cpu_flags)

    if args.toolchain == "all":
        if not toolchains:
            print(f"Error: No toolchains defined for system '{args.system}'.")
            sys.exit(1)
        for tc in sorted(toolchains):
            print(f"=== Compiling {args.system} / {tc} ===")
            compile_system(args.system, tc, sys_conf, toolchains[tc], args.scheduler, args.likwid)
            print(f"=== Submitting {args.system} / {tc} ===")
            if args.likwid:
                run_likwid_profiling(args.system, tc, sys_conf, toolchains[tc], args.scheduler, selected_cpu_flags)
            else:
                run_sweep(args.system, tc, sys_conf, toolchains[tc], args)
        return

    if args.toolchain not in toolchains:
        print(f"Error: Toolchain '{args.toolchain}' not defined for system '{args.system}'.")
        print(f"Known toolchains: {', '.join(toolchains)}")
        sys.exit(1)

    print(f"=== Compiling {args.system} / {args.toolchain} ===")
    compile_system(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args.scheduler, args.likwid)
    if args.likwid:
        run_likwid_profiling(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args.scheduler, selected_cpu_flags)
    else:
        run_sweep(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args)


if __name__ == "__main__":
    main()
