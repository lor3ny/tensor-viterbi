#!/usr/bin/env python3
"""print_status.py — Benchmark experiment coverage at a glance.

Rows: system/toolchain pairs known from systems.conf
Cols: T (number of timesteps): 1 k, 10 k, 100 k, 1 M
Cell: FULL | <found>/<expected> | NONE
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_ROOT  = "results"
SYSTEMS_CONF  = "systems.conf"

STATES    = [10, 15, 25, 50, 75]
DURATIONS = [100, 250, 500, 1000]
T_VALUES  = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

# T=10M uses a single fixed configuration (one data file exists) and is GPU-only.
T_OVERRIDES = {
    10_000_000: {"states": [100], "durations": [10_000], "gpu_only": True},
}

# Systems to hide from the summary table (still accessible via --system).
EXCLUDED_SYSTEMS = [
    "epyc-7763-bigmem",
    "epyc-9474f",
]

CPU_FUNCTIONS = [
    "HSMMLearn_CPP",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp_opt",
]
GPU_FUNCTIONS = ["decode_tensor_viterbi_cuda"]

# GPU generation order (older → newer)
GPU_GENERATION = {
    "a100":         0,   # A100  SM80  ~2020
    "mi250x":       1,   # MI250X gfx90a ~2021
    "h100":         2,   # H100  SM90  ~2022
    "gh200-hopper": 3,   # H100  SM90 (GH200) ~2023
    "mi300x":       4,   # MI300X gfx942 ~2023
}

# ── ANSI colours ──────────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code, text):
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
RED    = lambda t: _c("91", t)
BOLD   = lambda t: _c("1",  t)

# ── Layout constants ──────────────────────────────────────────────────────────

ROW_W  = 26   # visible width of row label
CELL_W = 11   # visible width of each cell ("T=1,000,000" = 11 chars)
LINE_W = ROW_W + (CELL_W + 2) * len(T_VALUES)

# ── Parse systems.conf ────────────────────────────────────────────────────────

def parse_systems_conf(path):
    """Return (sys_type dict, set of 'sys/tc' strings) from systems.conf."""
    # Strip commented lines so disabled system entries are ignored
    raw = Path(path).read_text()
    text = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))

    # SYS_TYPE[<sys>]="cpu"|"gpu"
    sys_type = {
        m.group(1): m.group(2)
        for m in re.finditer(r'SYS_TYPE\[([\w-]+)\]\s*=\s*"(\w+)"', text)
    }

    # SYS_MODULES[<sys>/<tc>]  — SYS_MODULES_BUILD has _BUILD so won't match
    sys_tc = {
        f"{m.group(1)}/{m.group(2)}"
        for m in re.finditer(r'SYS_MODULES\[([\w-]+)/([\w-]+)\]', text)
    }

    return sys_type, sys_tc

# ── Row ordering ──────────────────────────────────────────────────────────────

def row_key(sys_tc, sys_type):
    sys = sys_tc.split("/")[0]
    is_gpu = sys_type.get(sys, "cpu") == "gpu"
    gen = GPU_GENERATION.get(sys, 99) if is_gpu else 0
    return (1 if is_gpu else 0, gen, sys_tc)

# ── Coverage counting ─────────────────────────────────────────────────────────

def count_csvs(sys_tc, T, functions, states=None, durations=None):
    """Return (found, expected) CSV file counts."""
    root = Path(RESULTS_ROOT) / sys_tc
    _states = states if states is not None else STATES
    _durs   = durations if durations is not None else DURATIONS
    expected = len(_states) * len(_durs) * len(functions)
    found = sum(
        (root / f"{s}s_{d}d_{T}t_{fn}.csv").exists()
        for s in _states
        for d in _durs
        for fn in functions
    )
    return found, expected

# ── Cell formatting ───────────────────────────────────────────────────────────

def fmt_cell(found, expected):
    if found == 0:
        text, colorize = "NONE", RED
    elif found >= expected:
        text, colorize = "FULL", GREEN
    else:
        text, colorize = f"{found}/{expected}", YELLOW
    return colorize(text.rjust(CELL_W))


# ── Detail view ───────────────────────────────────────────────────────────────

def print_missing(sys_tc, sys_type):
    """Print every missing (N, D, T, function) combination for a system/toolchain."""
    sys = sys_tc.split("/")[0]
    is_gpu = sys_type.get(sys, "cpu") == "gpu"
    functions = GPU_FUNCTIONS if is_gpu else CPU_FUNCTIONS
    root = Path(RESULTS_ROOT) / sys_tc

    missing = []
    for T in T_VALUES:
        ov = T_OVERRIDES.get(T, {})
        if ov.get("gpu_only") and not is_gpu:
            continue
        _states = ov.get("states", STATES)
        _durs   = ov.get("durations", DURATIONS)
        for s in _states:
            for d in _durs:
                for fn in functions:
                    if not (root / f"{s}s_{d}d_{T}t_{fn}.csv").exists():
                        missing.append((s, d, T, fn))

    total_expected = sum(
        len(ov.get("states", STATES)) * len(ov.get("durations", DURATIONS)) * len(functions)
        for T in T_VALUES
        for ov in [T_OVERRIDES.get(T, {})]
        if not (ov.get("gpu_only") and not is_gpu)
    )
    total_missing  = len(missing)
    total_found    = total_expected - total_missing

    print(f"\n{BOLD(sys_tc)}  —  {total_found}/{total_expected} CSVs present\n")

    if total_missing == 0:
        print(GREEN("  All experiments complete.\n"))
        return

    # Group by T for readability
    from itertools import groupby
    missing.sort(key=lambda x: (x[2], x[0], x[1], x[3]))
    for T, group in groupby(missing, key=lambda x: x[2]):
        items = list(group)
        print(YELLOW(f"  T = {T:,}  ({len(items)} missing):"))
        for s, d, _, fn in items:
            print(f"    N={s:>2}  D={d:>4}  {fn}")
        print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark coverage status")
    parser.add_argument("--system", "-s", default=None,
                        help="system/toolchain to inspect in detail (e.g. epyc-7763-bigmem/cray)")
    args = parser.parse_args()

    # Always run relative to the repo root (where systems.conf lives)
    os.chdir(Path(__file__).parent)

    sys_type, conf_sys_tc = parse_systems_conf(SYSTEMS_CONF)

    # Include any result dirs not declared in systems.conf (shouldn't happen, but safe)
    result_sys_tc = {
        f"{d.parent.name}/{d.name}"
        for d in Path(RESULTS_ROOT).glob("*/*") if d.is_dir()
    }
    all_sys_tc = sorted(
        conf_sys_tc | result_sys_tc,
        key=lambda s: row_key(s, sys_type),
    )
    EXCLUDED_SYS_TC = {"xeon8480/gnu"}
    all_sys_tc = [s for s in all_sys_tc
                  if s.split("/")[0] not in EXCLUDED_SYSTEMS
                  and s not in EXCLUDED_SYS_TC]

    # ── Detail mode ────────────────────────────────────────────────────────────
    if args.system:
        if args.system not in all_sys_tc:
            sys.exit(f"Error: '{args.system}' not found. Known: {sorted(all_sys_tc)}")
        print_missing(args.system, sys_type)
        return

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "─" * LINE_W
    t_headers = [f"T={t:,}".rjust(CELL_W) for t in T_VALUES]
    header = f"{'System / Toolchain':<{ROW_W}}" + "".join(f"  {h}" for h in t_headers)

    print(sep)
    print(BOLD(header))
    print(sep)

    # ── Rows ──────────────────────────────────────────────────────────────────
    prev_sys = None
    for sys_tc in all_sys_tc:
        sys = sys_tc.split("/")[0]
        is_gpu = sys_type.get(sys, "cpu") == "gpu"
        functions = GPU_FUNCTIONS if is_gpu else CPU_FUNCTIONS

        # Blank line between different physical systems for readability
        if prev_sys is not None and sys != prev_sys:
            print()
        prev_sys = sys

        cells = []
        for T in T_VALUES:
            ov = T_OVERRIDES.get(T, {})
            if ov.get("gpu_only") and not is_gpu:
                cells.append(" " * CELL_W)
            else:
                cells.append(fmt_cell(*count_csvs(sys_tc, T, functions,
                                                   ov.get("states"), ov.get("durations"))))

        print(f"{sys_tc:<{ROW_W}}", end="")
        for c in cells:
            print(f"  {c}", end="")
        print()

    # ── Footer ────────────────────────────────────────────────────────────────
    print(sep)
    cpu_exp = len(STATES) * len(DURATIONS) * len(CPU_FUNCTIONS)
    gpu_exp = len(STATES) * len(DURATIONS) * len(GPU_FUNCTIONS)
    _10m_ov = T_OVERRIDES[10_000_000]
    gpu_10m_exp = len(_10m_ov["states"]) * len(_10m_ov["durations"]) * len(GPU_FUNCTIONS)
    print(
        f"\nExpected per T (≤1M):  "
        f"CPU = {cpu_exp} CSVs "
        f"({len(STATES)} states × {len(DURATIONS)} durations × {len(CPU_FUNCTIONS)} functions)   "
        f"GPU = {gpu_exp} CSVs ({len(GPU_FUNCTIONS)} function)\n"
        f"Expected T=10M (GPU only): {gpu_10m_exp} CSV "
        f"(N={_10m_ov['states']}, D={_10m_ov['durations']})\n"
        f"{GREEN('FULL')}  all expected files present   "
        f"{YELLOW('n/N')}  partial   "
        f"{RED('NONE')}  nothing yet"
    )


if __name__ == "__main__":
    main()
