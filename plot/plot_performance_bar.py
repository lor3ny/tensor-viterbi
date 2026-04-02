#!/usr/bin/env python3
"""
plot_performance_bar.py — speedup bar chart vs HSMMLearn C++ across all systems.

Speedup = mean_elapsed[HSMMLearn_CPP, ref] / mean_elapsed[function, system]
  - CPU systems: reference = same system's HSMMLearn_CPP
  - GPU systems: reference = slowest CPU HSMMLearn_CPP for that (D, T)
    (override with --ref-system)

X-axis  : (D, T) pairs; ticks labelled by D, grouped by T with bracket below.
Bars    : sorted by system/toolchain, then by function within each system.
Legend  : "Function name — system/toolchain"
Output  : plot/bars/<N>s_performance_bars.png

Usage:
  python plot/plot_performance_bar.py --states 75
  python plot/plot_performance_bar.py --states 75 --ref-system xeon8480/intel
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars")

FUNCTION_ORDER = [
    "HSMMLearn_CPP",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp",
    "decode_tensor_viterbi_cuda",
]
FUNCTION_LABELS = {
    "HSMMLearn_CPP":             "HSMMLearn C++",
    "HSMMLearn_OMP":             "HSMMLearn OMP",
    "decode_tensor_viterbi_cpp": "Tensor C++",
    "decode_tensor_viterbi_omp": "Tensor OMP",
    "decode_tensor_viterbi_cuda":"Tensor CUDA",
}


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    return sorted({
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        )
        for p in glob.glob(pattern)
    })


def load_means(system, n, d, t):
    """Return {function -> {mean, std}} (warmup iteration excluded)."""
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*.csv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["iteration"] != 0]
    result = {}
    for func, grp in df.groupby("function"):
        result[func] = {"mean": grp["elapsed_s"].mean(), "std": grp["elapsed_s"].std(ddof=1)}
    return result


def draw_group_bracket(ax, x_left, x_right, label, y_frac=-0.18, tick_h=0.03):
    """Bracket below x-axis spanning data x-coords x_left..x_right (y in axes fraction)."""
    trans = ax.get_xaxis_transform()
    for ln in [
        Line2D([x_left,  x_right], [y_frac,           y_frac],          transform=trans, color="black", lw=0.9, clip_on=False),
        Line2D([x_left,  x_left],  [y_frac - tick_h,  y_frac],          transform=trans, color="black", lw=0.9, clip_on=False),
        Line2D([x_right, x_right], [y_frac - tick_h,  y_frac],          transform=trans, color="black", lw=0.9, clip_on=False),
    ]:
        ax.add_line(ln)
    ax.text((x_left + x_right) / 2, y_frac - tick_h - 0.02, label,
            transform=trans, ha="center", va="top", fontsize=9, fontstyle="italic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",     "-s", type=int, required=True)
    parser.add_argument("--ref-system", default=None,
                        help="System/toolchain to use as HSMMLearn C++ reference for GPU bars.")
    args = parser.parse_args()
    N = args.states

    all_systems = discover_systems()
    if not all_systems:
        print("Error: no results found.", file=sys.stderr)
        sys.exit(1)

    # Discover all (D, T) pairs for this N
    dt_pairs = set()
    for sys_tc in all_systems:
        for f in glob.glob(os.path.join(RESULTS_ROOT, sys_tc, f"{N}s_*.csv")):
            fname = os.path.basename(f)
            parts = fname.split("_")
            try:
                d_val = int(parts[1].rstrip("d"))
                t_val = int(parts[2].rstrip("t"))
                dt_pairs.add((d_val, t_val))
            except (IndexError, ValueError):
                pass

    if not dt_pairs:
        print(f"Error: no data for N={N}.", file=sys.stderr)
        sys.exit(1)

    t_values = sorted({t for _, t in dt_pairs})
    d_values = sorted({d for d, _ in dt_pairs})

    # Load all data: all_data[sys_tc][(D, T)] = {func: {mean, std}}
    all_data = {}
    for sys_tc in all_systems:
        all_data[sys_tc] = {}
        for D_val, T_val in dt_pairs:
            all_data[sys_tc][(D_val, T_val)] = load_means(sys_tc, N, D_val, T_val)

    # System ordering: CPU first, then GPU
    cpu_sys = sorted(s for s in all_systems if "cuda" not in s and "mi250x" not in s)
    gpu_sys = sorted(s for s in all_systems if s not in cpu_sys)
    ordered_systems = cpu_sys + gpu_sys

    # Build sorted combo list: (func, sys_tc), ordered by system then function
    combo_set = set()
    for sys_tc in all_systems:
        for dt, funcs in all_data[sys_tc].items():
            for func in funcs:
                combo_set.add((func, sys_tc))

    def combo_key(c):
        func, sys_tc = c
        si = ordered_systems.index(sys_tc) if sys_tc in ordered_systems else len(ordered_systems)
        fi = FUNCTION_ORDER.index(func) if func in FUNCTION_ORDER else len(FUNCTION_ORDER)
        return (si, fi)

    ordered_combos = sorted(combo_set, key=combo_key)
    n_combos = len(ordered_combos)

    # Colors: tab10 cycles for up to 10 combos, tab20 for more
    cmap_name = "tab10" if n_combos <= 10 else "tab20"
    cmap  = plt.colormaps[cmap_name]
    n_map = 10 if cmap_name == "tab10" else 20
    colors = {combo: cmap(i / n_map) for i, combo in enumerate(ordered_combos)}

    # Reference lookup
    def get_ref(sys_tc, D_val, T_val):
        d = all_data[sys_tc].get((D_val, T_val), {})
        if "HSMMLearn_CPP" in d:
            return d["HSMMLearn_CPP"]["mean"]
        if args.ref_system:
            d2 = all_data.get(args.ref_system, {}).get((D_val, T_val), {})
            if "HSMMLearn_CPP" in d2:
                return d2["HSMMLearn_CPP"]["mean"]
        # Auto: slowest CPU HSMMLearn_CPP (most conservative for GPU)
        candidates = [
            all_data[s][(D_val, T_val)]["HSMMLearn_CPP"]["mean"]
            for s in all_systems
            if "HSMMLearn_CPP" in all_data[s].get((D_val, T_val), {})
        ]
        return max(candidates) if candidates else None

    # ── X-axis layout ────────────────────────────────────────────────────────
    # Groups: T values (sorted); within each group: D values (sorted)
    # Extra gap between T groups
    TICK_SPACING = 1.0
    GROUP_GAP    = 1.5   # extra space on top of TICK_SPACING between groups
    bar_width    = 0.8 / n_combos

    tick_pos = {}   # (T, D) -> x-center
    x = 0.0
    for ti, T_val in enumerate(t_values):
        if ti > 0:
            x += GROUP_GAP
        for D_val in d_values:
            tick_pos[(T_val, D_val)] = x
            x += TICK_SPACING

    # ── Draw ────────────────────────────────────────────────────────────────
    fig_width = max(16, x * 2.5 + 6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for ci, (func, sys_tc) in enumerate(ordered_combos):
        offset = (ci - (n_combos - 1) / 2) * bar_width
        xs       = []
        heights  = []
        errs     = []
        for T_val in t_values:
            for D_val in d_values:
                xs.append(tick_pos[(T_val, D_val)] + offset)
                fdata = all_data[sys_tc].get((D_val, T_val), {}).get(func)
                ref   = get_ref(sys_tc, D_val, T_val)
                if fdata is None or ref is None or fdata["mean"] == 0:
                    heights.append(0.0)
                    errs.append(0.0)
                else:
                    m = fdata["mean"]
                    s = fdata["std"]
                    heights.append(ref / m)
                    errs.append(ref * s / (m ** 2))

        label = f"{FUNCTION_LABELS.get(func, func)} — {sys_tc}"
        ax.bar(xs, heights, width=bar_width, color=colors[(func, sys_tc)],
               label=label, yerr=errs, capsize=2,
               error_kw={"elinewidth": 0.7, "ecolor": "black"}, zorder=2)

    # Horizontal reference line at 1×
    ax.axhline(1.0, color="black", lw=0.9, linestyle="--", zorder=3)

    # X ticks: one per (T, D), labelled by D only
    all_ticks = [(T_val, D_val) for T_val in t_values for D_val in d_values]
    ax.set_xticks([tick_pos[(T, D)] for T, D in all_ticks])
    ax.set_xticklabels([f"D = {D}" for _, D in all_ticks], fontsize=8)

    # T-group brackets below x-axis
    half_group = (n_combos / 2) * bar_width
    for T_val in t_values:
        x_left  = tick_pos[(T_val, d_values[0])]  - half_group - 0.05
        x_right = tick_pos[(T_val, d_values[-1])] + half_group + 0.05
        draw_group_bracket(ax, x_left, x_right, f"T = {T_val:,}")

    ax.set_ylabel("Speedup vs HSMMLearn C++  (higher = faster)", fontsize=9)
    ref_note = f"GPU ref: {args.ref_system}" if args.ref_system else "GPU ref: slowest CPU HSMMLearn C++ (conservative)"
    ax.set_title(f"Speedup vs HSMMLearn C++  —  N = {N} states    ({ref_note})", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, title="function — system/toolchain", title_fontsize=8)

    plt.subplots_adjust(bottom=0.22)
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{N}s_performance_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
