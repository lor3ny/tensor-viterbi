"""bench status — combine the manifest(s), the results directory, and (on
SLURM) `squeue` output into a done/running/pending/failed report.
"""

from __future__ import annotations

import subprocess

from .execution import check_job_complete, results_dir_for
from .manifest import read_manifest
from .paths import RUNS_DIR


def _query_squeue() -> dict[str, str]:
    """One bulk `squeue --me` call mapped by job name, equivalent to (but far
    cheaper than) calling `squeue --me --name=tv_<stem>` once per job."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%j %T"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return {}
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    return mapping


def all_manifests(system: str) -> list[tuple[str, list[dict]]]:
    system_dir = RUNS_DIR / system
    if not system_dir.exists():
        return []
    out = []
    for path in sorted(system_dir.glob("*.jsonl")):
        out.append((path.stem, read_manifest(path)))
    return out


def job_status(job: dict, scheduler: str, squeue: dict[str, str]) -> tuple[str, str]:
    results_dir = results_dir_for(job["system"], job["toolchain"])
    status, detail = check_job_complete(job, results_dir)
    if status in ("pending", "failed") and scheduler == "slurm":
        state = squeue.get(f"tv_{job['stem']}")
        if state == "PENDING":
            return "pending", "queued"
        if state == "RUNNING":
            return "running", "in squeue"
    return status, detail


def print_status(system: str, scheduler: str) -> None:
    manifests = all_manifests(system)
    if not manifests:
        print(f"No manifest found for system '{system}' under {RUNS_DIR / system}. "
              f"Run `bench plan --system {system} --pack <pack>` first.")
        return

    squeue = _query_squeue() if scheduler == "slurm" else {}
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for pack, jobs in manifests:
        print(f"\n=== pack: {pack} ({len(jobs)} job(s)) ===")
        for job in jobs:
            status, detail = job_status(job, scheduler, squeue)
            counts[status] += 1
            print(f"  {job['toolchain']}/{job['stem']}: {status}  [{detail}]")

    total = sum(counts.values())
    print(f"\nSummary: {total} job(s) — "
          f"done={counts['done']} running={counts['running']} "
          f"pending={counts['pending']} failed={counts['failed']}")
