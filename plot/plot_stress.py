#!/usr/bin/env python3
"""
plot_stress.py — absolute execution time for the stress-test (T=10M) configuration.

Only GPU data is available for T=10M (N=100, D=10000).
Plots a simple horizontal bar chart with one bar per GPU system,
showing mean elapsed time in seconds (with min/max error bars).

Output: bars/stress/stress_100s_10000000t.pdf

Usage:
  python plot/plot_stress.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars", "stress")

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "h200":      "cuda",
    "mi250x":    "cray",
    "mi300x":    "amd",
    "gh200-hopper": "gnu",
    "gh200-grace":  "gnu14",
}

SYSTEM_LABELS = {
    "a100":         "A100",
    "h100":         "H100",
    "h200":          "H200",
    "gh200-hopper": "H100",
    "b200":         "B200",
    "mi250x":       "MI250X",
    "mi300x":       "MI300X",
}

# Display order (fastest-expected to slowest)
GPU_ORDER = ["b200", "mi300x", "h100", "h200", "a100", "mi250x"]

GPU_COLORS = {
    "a100":         "#4393C3",
    "h100":         "#74C476",
    "gh200-hopper": "#FD8D3C",
    "h200":         "#4DAF4A",
    "b200":         "#D6604D",
    "mi250x":       "#9E9AC8",
    "mi300x":       "#E7298A",
}

N, D, T = 100, 10000, 10000000
FUNC    = "decode_tensor_viterbi_cuda"


def load_data():
    """Return dict: system -> (mean_s, min_s, max_s)."""
    records = {}
    for system, toolchain in DEFAULT_TOOLCHAINS.items():
        pattern = os.path.join(
            RESULTS_ROOT, system, toolchain,
            f"{N}s_{D}d_{T}t_{FUNC}.csv",
        )
        files = glob.glob(pattern)
        if not files:
            continue
        try:
            df = pd.read_csv(files[0])
        except Exception:
            continue
        if "elapsed_s" not in df.columns:
            continue
        vals = df["elapsed_s"].dropna().values
        if len(vals) == 0:
            continue
        if len(vals) > 1:
            vals = vals[1:]  # skip first iteration as warmup
        records[system] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
    return records


def make_plot(data):
    # Filter to systems with data, in display order
    systems = [s for s in GPU_ORDER if s in data]
    if not systems:
        print("No data found.")
        return

    # Sort slowest -> fastest (slowest at left of vertical bar chart)
    order  = np.argsort([-data[s][0] for s in systems])
    systems = [systems[i] for i in order]
    means  = np.array([data[s][0] for s in systems])
    mins   = np.array([data[s][1] for s in systems])
    maxs   = np.array([data[s][2] for s in systems])
    errs_lo = means - mins
    errs_hi = maxs - means
    labels  = [SYSTEM_LABELS.get(s, s) for s in systems]
    colors  = [GPU_COLORS.get(s, "#aaaaaa") for s in systems]

    means_min = means / 60
    mins_min  = mins  / 60
    maxs_min  = maxs  / 60
    errs_lo_min = means_min - mins_min
    errs_hi_min = maxs_min - mins_min

    fig, ax = plt.subplots(figsize=(6, 5.5))

    x_pos = np.arange(len(systems))
    bars = ax.bar(
        x_pos, means_min,
        yerr=[errs_lo_min, errs_hi_min],
        color=colors,
        edgecolor="#444", linewidth=0.7,
        error_kw=dict(elinewidth=1.2, capsize=4, ecolor="#333"),
        width=0.55,
    )

    # Annotate values above bars
    _gap = max(means_min) * 0.02
    for i, m in enumerate(means_min):
        hrs = m / 60
        lbl = f"{hrs:.1f}h" if hrs >= 1.0 else f"{m:.1f}min"
        ax.text(x_pos[i], m + _gap, lbl,
                va="bottom", ha="center", fontsize=13, rotation=90)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylabel("Runtime (minutes)", fontsize=14)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Extra top margin for annotations
    ax.set_ylim(0, max(means_min) * 1.25)



    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.22)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"stress_{N}s_{T}t.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    data = load_data()
    if not data:
        print("No T=10M GPU data found.")
        return
    print(f"Found data for: {', '.join(data.keys())}")
    make_plot(data)


if __name__ == "__main__":
    main()
