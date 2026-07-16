"""Runs every plot/*.py plotter against results/. Shared by `bench plot` and
plot/plot_all.py so there is a single list of "which plotters exist".
"""

from __future__ import annotations

import subprocess
import sys

from .paths import RESULTS_DIR, SCRIPT_DIR

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


def _discover_likwid_targets() -> list[tuple[str, str]]:
    """Every results/<system>/<toolchain>/ dir that has LIKWID CSVs."""
    if not RESULTS_DIR.is_dir():
        return []
    targets = []
    for sys_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        for tc_dir in sorted(p for p in sys_dir.iterdir() if p.is_dir()):
            if any(tc_dir.glob("likwid_*.csv")):
                targets.append((sys_dir.name, tc_dir.name))
    return targets


def run_all(all_toolchains: bool = False, system: str | None = None,
            toolchain: str | None = None) -> bool:
    """Runs every always-on plotter, plus plot_likwid.py for each system/
    toolchain with LIKWID data (all of them if all_toolchains, else just the
    given system+toolchain). Returns True iff every invoked plotter exited 0."""
    extra = ["--all-toolchains"] if all_toolchains else []

    failed: list[str] = []
    for script_name in ALWAYS_PLOTTERS:
        if not _run(script_name, extra):
            failed.append(script_name)

    if system and toolchain:
        likwid_targets = [(system, toolchain)]
    elif all_toolchains:
        likwid_targets = _discover_likwid_targets()
        if not likwid_targets:
            print("\n[plot_likwid.py] skipped — no LIKWID CSV data found under results/")
    else:
        likwid_targets = []
        print("\n[plot_likwid.py] skipped — pass --system and --toolchain, "
              "or --all-toolchains, to include it")

    for sys_name, tc_name in likwid_targets:
        likwid_extra = ["--system", sys_name, "--toolchain", tc_name]
        if not _run("plot_likwid.py", likwid_extra):
            failed.append(f"plot_likwid.py ({sys_name}/{tc_name})")

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return False
    print("All plotters completed successfully.")
    return True
