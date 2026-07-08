"""Shared paths and the repo-root working-directory check."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR      = Path(__file__).resolve().parent.parent
SYSTEMS_DIR     = SCRIPT_DIR / "systems"
RUNS_DIR        = SCRIPT_DIR / "runs"
RESULTS_DIR     = SCRIPT_DIR / "results"
DATA_DIR        = SCRIPT_DIR / "data"
WALLTIMES_FILE  = SCRIPT_DIR / "walltimes.yaml"
PARAMS_FILE     = SCRIPT_DIR / "benchmark_params.cfg"
RUN_ONE_SH      = SCRIPT_DIR / "run_one.sh"
RUN_ONE_SLURM   = SCRIPT_DIR / "run_one.slurm"
LIKWID_ONE_SH   = SCRIPT_DIR / "likwid_one.sh"
LIKWID_ONE_SLURM = SCRIPT_DIR / "likwid_one.slurm"


def require_repo_root(argv0: str, argv: list[str]) -> None:
    if Path.cwd().resolve() != SCRIPT_DIR:
        print("Error: must be run from the repository root.")
        print(f"  cd {SCRIPT_DIR} && {argv0} {' '.join(argv)}")
        sys.exit(1)


def require_python_version() -> None:
    if sys.version_info < (3, 10):
        print(f"Error: Python >= 3.10 required, found {sys.version.split()[0]}.")
        print("Activate a Python 3.10+ environment before running bench.")
        sys.exit(1)
