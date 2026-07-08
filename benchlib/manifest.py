"""Plan: walk the (states, duration, timesteps) grid, bucket by pack, and
write the job manifest consumed by `bench run`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import RUNS_DIR
from .params import (
    PACKS, load_benchmark_params, get_walltime, pack_of_walltime, hms_to_seconds,
    effective_iterations,
)


def build_jobs(system: str, toolchains: list[str], pack: str, viterbi_flags: str,
               iterations: int) -> tuple[list[dict], int]:
    """Returns (jobs, skipped_count)."""
    states, durations, timesteps = load_benchmark_params()
    jobs: list[dict] = []
    skipped = 0
    for toolchain in toolchains:
        for s in states:
            for d in durations:
                for t in timesteps:
                    iters = effective_iterations(t, iterations)
                    walltime = get_walltime(s, d, t, viterbi_flags, iters)
                    if pack_of_walltime(walltime) != pack:
                        skipped += 1
                        continue
                    jobs.append({
                        "stem":       f"{s}s_{d}d_{t}t",
                        "system":     system,
                        "toolchain":  toolchain,
                        "states":     s,
                        "duration":   d,
                        "timesteps":  t,
                        "flags":      viterbi_flags,
                        "data_path":  f"data/{s}states_{t}steps_{d}dur.json",
                        "walltime":   walltime,
                        "iterations": iters,
                    })
    return jobs, skipped


def manifest_path(system: str, pack: str) -> Path:
    return RUNS_DIR / system / f"{pack}.jsonl"


def write_manifest(system: str, pack: str, jobs: list[dict]) -> Path:
    path = manifest_path(system, pack)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    return path


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    jobs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            jobs.append(json.loads(line))
    return jobs


def total_walltime_hours(jobs: list[dict]) -> float:
    return sum(hms_to_seconds(j["walltime"]) for j in jobs) / 3600.0


def print_preview(system: str, pack: str, jobs: list[dict], scheduler: str) -> None:
    lo, hi = PACKS[pack]
    print(f"Pack selected: {pack} ({lo // 3600}h–{hi // 3600}h walltime range)")
    print(f"Plan: {len(jobs)} job(s) for system '{system}'.")
    if scheduler == "local":
        hours = total_walltime_hours(jobs)
        print(f"Estimated total serial walltime (local, sequential): {hours:.1f}h")
        print("  Local runs execute one job at a time. Use --jobs A-B to run a slice, "
              "or --max-hours H to stop once the cumulative estimate would exceed H.")
