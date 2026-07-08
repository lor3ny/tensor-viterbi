"""bench likwid — LIKWID hardware-counter profiling. CPU-only, fixed data
file, no pack. Both backends run the identical likwid_one.sh (see
likwid_one.sh/likwid_one.slurm) instead of the old duplicated
generate_likwid_slrm()/run_likwid_local() pair.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import flags as flagslib
from .execution import build_sbatch_flags, results_dir_for
from .paths import SCRIPT_DIR, LIKWID_ONE_SH, LIKWID_ONE_SLURM


def _base_env(conf: dict, toolchain: str) -> dict:
    tc_conf = conf["toolchains"][toolchain]
    return {
        "SYS_NAME":       f"{conf['name']}/{toolchain}",
        "SYS_MODULES":    tc_conf.get("modules", ""),
        "SYS_UENV":       tc_conf.get("uenv", ""),
        "SYS_OMP_BIND":   conf.get("omp_bind", ""),
        "SYS_OMP_PLACES": conf.get("omp_places", ""),
        "SYS_CPUS":       str(conf.get("cpus", "")),
        "LIKWID_GROUPS":        " ".join(flagslib.LIKWID_PERF_GROUPS),
    }


def run_likwid_local(conf: dict, toolchain: str, cpu_flags: list[str] | None) -> None:
    results_dir = results_dir_for(conf["name"], toolchain)
    results_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        **_base_env(conf, toolchain),
        "DATA_PATH":            str(SCRIPT_DIR / flagslib.LIKWID_DATA_LOCAL),
        "RESULTS_DIR":          str(results_dir),
        "LIKWID_VERSION_FLAGS": " ".join(cpu_flags or flagslib.LIKWID_CPU_FLAGS),
    }
    print(f"LIKWID profiling: {conf['name']}/{toolchain} (local)")
    out_path = results_dir / "likwid.out"
    err_path = results_dir / "likwid.err"
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        subprocess.run(["bash", str(LIKWID_ONE_SH)], env=env, stdout=fout, stderr=ferr,
                        cwd=str(SCRIPT_DIR))


def submit_likwid_slurm(conf: dict, toolchain: str, cpu_flags: list[str] | None) -> None:
    results_dir = results_dir_for(conf["name"], toolchain)
    results_dir.mkdir(parents=True, exist_ok=True)
    job_env = {
        **_base_env(conf, toolchain),
        "DATA_PATH":            str(SCRIPT_DIR / flagslib.LIKWID_DATA_SLURM),
        "RESULTS_DIR":          str(results_dir),
        "LIKWID_VERSION_FLAGS": " ".join(cpu_flags or flagslib.LIKWID_CPU_FLAGS),
    }
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in job_env.items())
    stem = "likwid"
    cmd = [
        "sbatch", *build_sbatch_flags(conf),
        f"--export={export_str}",
        f"--job-name=tv_{stem}",
        "--time=02:00:00",
        f"--output={results_dir / (stem + '.out')}",
        f"--error={results_dir / (stem + '.err')}",
        str(LIKWID_ONE_SLURM),
    ]
    print(f"  -> Submitting LIKWID profiling job for {conf['name']}/{toolchain}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)


def run_likwid(conf: dict, toolchain: str, scheduler: str, cpu_flags: list[str] | None) -> None:
    if conf["type"] != "cpu":
        print(f"Warning: LIKWID profiling is CPU-only (system type={conf['type']}). "
              f"Skipping {conf['name']}/{toolchain}.")
        return
    if scheduler == "local":
        run_likwid_local(conf, toolchain, cpu_flags)
    else:
        submit_likwid_slurm(conf, toolchain, cpu_flags)
