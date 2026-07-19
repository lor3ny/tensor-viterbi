"""Shared job execution: sbatch flags/env (identical to the old
_build_sbatch_flags/_build_job_env), the local/slurm dispatch calls (both
invoke run_one.sh — see run_one.sh/run_one.slurm), resume/skip detection,
and the `bench run` manifest driver.
"""

import subprocess
import sys
import time
from pathlib import Path

from . import flags as flagslib
from .paths import SCRIPT_DIR, RESULTS_DIR, RUN_ONE_SH, RUN_ONE_SLURM


def results_dir_for(system: str, toolchain: str) -> Path:
    return RESULTS_DIR / system / toolchain


def build_sbatch_flags(conf: dict) -> list[str]:
    slurm = conf["slurm"]
    out: list[str] = []
    if slurm.get("account"):
        out.append(f"--account={slurm['account']}")
    if slurm.get("partition"):
        out.append(f"--partition={slurm['partition']}")
    if slurm.get("qos"):
        out.append(f"--qos={slurm['qos']}")
    out.append("--gres=gpu:1" if conf["type"] == "gpu" else f"--cpus-per-task={conf.get('cpus', 1)}")
    return out


def build_job_env(conf: dict, toolchain: str, job: dict, results_dir: Path,
                   nsys: bool, ncu: bool) -> dict:
    tc_conf = conf["toolchains"][toolchain]
    return {
        "SYS_NAME":             f"{conf['name']}/{toolchain}",
        "SYS_TYPE":             conf["type"],
        "SYS_MODULES":          tc_conf.get("modules", ""),
        "SYS_METRICS_BACKEND":  tc_conf.get("metrics_backend", ""),
        "SYS_UENV":             tc_conf.get("uenv", ""),
        "SYS_OMP_BIND":         conf.get("omp_bind", ""),
        "SYS_OMP_PLACES":       conf.get("omp_places", ""),
        "SYS_CPUS":             str(conf.get("cpus", "")),
        "VITERBI_FLAGS":        job["flags"],
        "BENCHMARK_ITERATIONS": str(job["iterations"]),
        "DATA_PATH":            str(SCRIPT_DIR / job["data_path"]),
        "NSYS_PROFILE":         "1" if nsys else "0",
        "NCU_PROFILE":          "1" if ncu else "0",
        "RESULTS_DIR":          str(results_dir),
        "JOB_STEM":             job["stem"],
    }


# ── Resume / completeness check ──────────────────────────────────────────
#
# A successful viterbi_app.py run writes one "<stem>_<fname>.csv" per
# requested backend (header row + one row per iteration; see _bench()/
# _bench_baseline() in viterbi_app.py) and, on an uncaught exception, a
# Python traceback on stderr. We treat a job as "done" only if every CSV a
# successful run for its flags would produce exists with the full row
# count, and stderr contains no traceback. This only inspects artifacts
# viterbi_app.py already writes — no new marker files are introduced.
def check_job_complete(job: dict, results_dir: Path) -> tuple[str, str]:
    stem = job["stem"]
    out_path = results_dir / f"{stem}.out"
    err_path = results_dir / f"{stem}.err"
    if not out_path.exists() and not err_path.exists():
        return "pending", "not started"

    problems = []
    for fname in flagslib.expected_csv_stems(job["flags"]):
        csv_path = results_dir / f"{stem}_{fname}.csv"
        if not csv_path.exists():
            problems.append(f"{csv_path.name} missing")
            continue
        try:
            n_lines = sum(1 for _ in open(csv_path))
        except OSError:
            n_lines = 0
        want = job["iterations"] + 1  # header + one row per iteration
        if n_lines < want:
            problems.append(f"{csv_path.name} truncated ({n_lines}/{want} lines)")

    if err_path.exists():
        err_text = err_path.read_text(errors="replace")
        if "Traceback (most recent call last)" in err_text:
            problems.append("traceback in .err")

    if problems:
        return "failed", "; ".join(problems)
    return "done", "complete"


# ── Dispatch ──────────────────────────────────────────────────────────────

def dispatch_local(job: dict, conf: dict, results_dir: Path, nsys: bool, ncu: bool) -> None:
    import os
    env = {**os.environ, **build_job_env(conf, job["toolchain"], job, results_dir, nsys, ncu)}
    print(f"  -> flags=[{job['flags']}] iterations={job['iterations']} (local)")
    out_path = results_dir / f"{job['stem']}.out"
    err_path = results_dir / f"{job['stem']}.err"
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        subprocess.run(["bash", str(RUN_ONE_SH)], env=env, stdout=fout, stderr=ferr, cwd=str(SCRIPT_DIR))


def dispatch_slurm(job: dict, conf: dict, results_dir: Path, sbatch_flags: list[str],
                    nsys: bool, ncu: bool) -> None:
    job_env = build_job_env(conf, job["toolchain"], job, results_dir, nsys, ncu)
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in job_env.items())
    stem = job["stem"]
    cmd = [
        "sbatch", *sbatch_flags,
        f"--export={export_str}",
        f"--job-name=tv_{stem}",
        f"--time={job['walltime']}",
        f"--output={results_dir / (stem + '.out')}",
        f"--error={results_dir / (stem + '.err')}",
        str(RUN_ONE_SLURM),
    ]
    print(f"  -> flags=[{job['flags']}] iterations={job['iterations']}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return
    time.sleep(0.1)


# ── Manifest driver (bench run) ──────────────────────────────────────────

def parse_jobs_slice(spec: str, n: int) -> tuple[int, int]:
    """'A-B' (1-indexed, inclusive) -> 0-indexed [start, end) for slicing."""
    a, _, b = spec.partition("-")
    lo = int(a)
    hi = int(b) if b else n
    lo = max(lo, 1)
    hi = min(hi, n)
    return lo - 1, hi


def run_manifest(jobs: list[dict], conf: dict, scheduler: str, *, force: bool,
                  only_failed: bool, jobs_slice: str | None, max_hours: float | None,
                  nsys: bool, ncu: bool, compile_fn) -> None:
    from .params import hms_to_seconds

    if jobs_slice:
        start, end = parse_jobs_slice(jobs_slice, len(jobs))
        jobs = jobs[start:end]

    sbatch_flags = build_sbatch_flags(conf) if scheduler == "slurm" else []
    compiled: set[str] = set()

    budget_seconds = max_hours * 3600 if max_hours is not None else None
    spent_seconds = 0
    remaining_after_budget: list[dict] = []

    for job in jobs:
        results_dir = results_dir_for(conf["name"], job["toolchain"])
        results_dir.mkdir(parents=True, exist_ok=True)
        status, detail = check_job_complete(job, results_dir)

        if not force:
            if only_failed:
                if status != "failed":
                    print(f"Skipping {job['stem']} ({job['toolchain']}): "
                          f"status={status} (--only-failed wants failed jobs) [{detail}]")
                    continue
            elif status == "done":
                print(f"Skipping {job['stem']} ({job['toolchain']}): already complete")
                continue

        if scheduler == "local" and budget_seconds is not None:
            job_seconds = hms_to_seconds(job["walltime"])
            if spent_seconds + job_seconds > budget_seconds:
                remaining_after_budget.append(job)
                continue
            spent_seconds += job_seconds

        if job["toolchain"] not in compiled:
            print(f"=== Compiling {conf['name']} / {job['toolchain']} ===")
            compile_fn(job["toolchain"])
            compiled.add(job["toolchain"])

        print(f"Submitting: System={conf['name']}, States={job['states']}, "
              f"Duration={job['duration']}, Timesteps={job['timesteps']}, "
              f"Walltime={job['walltime']}")

        if scheduler == "local":
            dispatch_local(job, conf, results_dir, nsys, ncu)
        else:
            dispatch_slurm(job, conf, results_dir, sbatch_flags, nsys, ncu)

    if remaining_after_budget:
        print(f"--max-hours reached: {len(remaining_after_budget)} job(s) not started "
              f"(re-run with a larger --max-hours, or without it, to continue):")
        for job in remaining_after_budget:
            print(f"  {job['toolchain']}/{job['stem']} (walltime {job['walltime']})")
