"""Runs every plot/*.py plotter against results/. Shared by `bench plot` and
plot/plot_all.py so there is a single list of "which plotters exist".
"""

from __future__ import annotations

import subprocess
import sys

from .paths import SCRIPT_DIR

PLOT_DIR = SCRIPT_DIR / "plot"

# Plotters that need no arguments beyond --all-toolchains (which they either
# use or, in plot_stress.py's case, silently ignore since it has no argparse).
ALWAYS_PLOTTERS = [
    "plot_sc.py",
    "plot_mc_gpu.py",
    "plot_speedup_heatmap.py",
    "plot_energy.py",
    "plot_stress.py",
]


def _run(script_name: str, extra: list[str]) -> bool:
    script = PLOT_DIR / script_name
    print(f"\n[{script_name}]")
    result = subprocess.run([sys.executable, str(script), *extra], cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"[{script_name}] FAILED (exit {result.returncode})")
        return False
    return True


def run_all(all_toolchains: bool = False, system: str | None = None,
            toolchain: str | None = None) -> bool:
    """Runs every always-on plotter, plus plot_likwid.py if system+toolchain
    are given. Returns True iff every invoked plotter exited 0."""
    extra = ["--all-toolchains"] if all_toolchains else []

    failed: list[str] = []
    for script_name in ALWAYS_PLOTTERS:
        if not _run(script_name, extra):
            failed.append(script_name)

    if system and toolchain:
        likwid_extra = ["--system", system, "--toolchain", toolchain]
        if not _run("plot_likwid.py", likwid_extra):
            failed.append("plot_likwid.py")
    else:
        print("\n[plot_likwid.py] skipped — pass --system and --toolchain to include it")

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return False
    print("All plotters completed successfully.")
    return True
