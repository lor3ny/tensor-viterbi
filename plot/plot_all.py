#!/usr/bin/env python3
"""
plot_all.py — run all available plotters in sequence.

Plotters that need no arguments are always executed.
plot_likwid.py requires --system and --toolchain; pass them here to include it.

Usage:
  python plot/plot_all.py
  python plot/plot_all.py --system xeon8480 --toolchain intel
"""

import argparse
import subprocess
import sys
from pathlib import Path

PLOT_DIR   = Path(__file__).resolve().parent
SCRIPT_DIR = PLOT_DIR.parent


def run(script: Path, extra: list[str] | None = None) -> bool:
    print(f"\n[{script.name}]")
    result = subprocess.run(
        [sys.executable, str(script), *(extra or [])],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"[{script.name}] FAILED (exit {result.returncode})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all tensor-viterbi plotters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system",    default=None,
                        help="System key for plot_likwid.py (required to run it)")
    parser.add_argument("--toolchain", default=None,
                        help="Toolchain key for plot_likwid.py (required to run it)")
    args = parser.parse_args()

    always = [
        PLOT_DIR / "plot_sc.py",
        PLOT_DIR / "plot_mc_gpu.py",
        PLOT_DIR / "plot_speedup_heatmap.py",
        PLOT_DIR / "plot_energy.py",
        PLOT_DIR / "plot_stress.py",
    ]

    failed: list[str] = []
    for script in always:
        if not run(script):
            failed.append(script.name)

    if args.system and args.toolchain:
        extra = ["--system", args.system, "--toolchain", args.toolchain]
        if not run(PLOT_DIR / "plot_likwid.py", extra):
            failed.append("plot_likwid.py")
    else:
        print("\n[plot_likwid.py] skipped — pass --system and --toolchain to include it")

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All plotters completed successfully.")


if __name__ == "__main__":
    main()
