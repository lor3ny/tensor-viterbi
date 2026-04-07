#!/usr/bin/env python3
"""
plot_speedup_heatmap.py — speedup heatmaps across system combinations.

Two heatmap types produced:
  1. Tensor OMP / HSMMLearn OMP  — one figure per CPU system
  2. Tensor CUDA / HSMMLearn OMP — one figure per (GPU system x CPU system) pair

Each figure has one panel per timestep value T.
Grid: columns = N (n_states, ascending), rows = D (max_duration, high->low).

Output: plot/heatmaps/<label>.png

Usage:
  python plot/plot_speedup_heatmap.py
  python plot/plot_speedup_heatmap.py --all-toolchains
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmaps")

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",   # matches epyc-7763-bigmem etc.
    "xeon8480":  "gnu",
}


def _filter_systems(all_systems, use_all):
    """Keep only the default toolchain per system prefix unless use_all is True."""
    if use_all:
        return all_systems
    kept = []
    for sys_tc in all_systems:
        system, toolchain = sys_tc.split("/", 1)
        default = next(
            (tc for prefix, tc in DEFAULT_TOOLCHAINS.items() if system.startswith(prefix)),
            None,
        )
        if default is None or toolchain == default:
            kept.append(sys_tc)
    return kept


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    return sorted({
        "/".join([
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        ])
        for p in glob.glob(pattern)
    })


def is_gpu(sys_tc):
    system = sys_tc.split("/")[0]
    return any(system.startswith(g) for g in ("a100", "h100", "mi250x"))


def load_means(sys_tc):
    """Return {(N, D, T, function) -> mean_elapsed_s}, warmup iteration 0 excluded."""
    result = {}
    for f in glob.glob(os.path.join(RESULTS_ROOT, sys_tc, "*.csv")):
        if "_metrics" in os.path.basename(f):
            continue
        try:
            df = pd.read_csv(f)
            df = df[df["iteration"] != 0]
            for (func, n, d, t), grp in df.groupby(
                ["function", "n_states", "max_duration", "timesteps"]
            ):
                key = (int(n), int(d), int(t), func)
                if key in result:
                    # Average across multiple files for the same key
                    result[key] = (result[key] + grp["elapsed_s"].mean()) / 2
                else:
                    result[key] = grp["elapsed_s"].mean()
        except Exception:
            pass
    return result


def make_heatmap(title, speedup_label, speedup_dict, out_path):
    """
    speedup_dict: {(N, D, T) -> speedup_value}
    Saves one figure with one panel per unique T value.
    """
    if not speedup_dict:
        print(f"  Skipped (no data): {title}")
        return

    all_n = sorted({n for n, d, t in speedup_dict})
    all_d = sorted({d for n, d, t in speedup_dict}, reverse=True)  # high -> low on Y-axis
    all_t = sorted({t for n, d, t in speedup_dict})

    fig, axes = plt.subplots(
        1, len(all_t),
        figsize=(4.5 * len(all_t), 4),
        squeeze=False,
    )
    axes = list(axes[0])

    all_vals = [v for v in speedup_dict.values() if not np.isnan(v)]
    if not all_vals:
        plt.close(fig)
        return
    vmin, vmax = min(all_vals), max(all_vals)
    # Diverging norm centred at 1.0: red = slower than reference, green = faster.
    _vmin = min(vmin, 0.95)   # guarantee vmin < vcenter
    _vmax = max(vmax, 1.05)   # guarantee vmax > vcenter
    norm = mcolors.TwoSlopeNorm(vmin=_vmin, vcenter=1.0, vmax=_vmax)
    cmap = "RdYlGn"

    for ax, T in zip(axes, all_t):
        matrix = np.full((len(all_d), len(all_n)), np.nan)
        for ri, D in enumerate(all_d):
            for ci, N in enumerate(all_n):
                v = speedup_dict.get((N, D, T))
                if v is not None:
                    matrix[ri, ci] = v

        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

        for ri in range(len(all_d)):
            for ci in range(len(all_n)):
                val = matrix[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{val:.1f}x", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="black")

        ax.set_xticks(range(len(all_n)))
        ax.set_xticklabels(all_n, fontsize=9)
        ax.set_yticks(range(len(all_d)))
        ax.set_yticklabels(all_d, fontsize=9)
        ax.set_xlabel("States (N)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Max Duration (D)", fontsize=10)
        ax.set_title(f"T = {T:,}", fontsize=11, fontweight="bold")

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, shrink=0.8, pad=0.02,
    ).set_label(f"Speedup {speedup_label}  (x)", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    os.makedirs(OUT_ROOT, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speedup heatmaps from benchmark results."
    )
    parser.add_argument(
        "--all-toolchains", action="store_true",
        help="Include all toolchains; default shows one toolchain per system.",
    )
    args = parser.parse_args()

    all_systems = _filter_systems(discover_systems(), args.all_toolchains)
    if not all_systems:
        print("No results found under results/.", file=sys.stderr)
        sys.exit(1)

    cpu_systems = [s for s in all_systems if not is_gpu(s)]
    gpu_systems = [s for s in all_systems if is_gpu(s)]

    print(f"CPU systems : {cpu_systems}")
    print(f"GPU systems : {gpu_systems}")

    data = {s: load_means(s) for s in all_systems}

    # 1. Tensor OMP / HSMMLearn OMP  (one figure per CPU system)
    print("\n-- Tensor OMP / HSMMLearn OMP --")
    for cpu_sys in cpu_systems:
        d = data[cpu_sys]
        speedup = {}
        for (n, dur, t, func), mean_s in d.items():
            if func == "decode_tensor_viterbi_omp" and mean_s > 0:
                ref = d.get((n, dur, t, "HSMMLearn_OMP"))
                if ref and ref > 0:
                    speedup[(n, dur, t)] = ref / mean_s
        slug = cpu_sys.replace("/", "_")
        make_heatmap(
            title=f"Tensor OMP / HSMMLearn OMP -- {cpu_sys}",
            speedup_label="Tensor OMP / HSMMLearn OMP",
            speedup_dict=speedup,
            out_path=os.path.join(OUT_ROOT, f"omp_vs_omp_{slug}.png"),
        )

    # 2. Tensor CUDA / HSMMLearn OMP  (one figure per GPU x CPU pair)
    print("\n-- Tensor CUDA / HSMMLearn OMP --")
    for gpu_sys in gpu_systems:
        for cpu_sys in cpu_systems:
            gpu_d = data[gpu_sys]
            cpu_d = data[cpu_sys]
            speedup = {}
            for (n, dur, t, func), mean_s in gpu_d.items():
                if func == "decode_tensor_viterbi_cuda" and mean_s > 0:
                    ref = cpu_d.get((n, dur, t, "HSMMLearn_OMP"))
                    if ref and ref > 0:
                        speedup[(n, dur, t)] = ref / mean_s
            gpu_slug = gpu_sys.replace("/", "_")
            cpu_slug = cpu_sys.replace("/", "_")
            make_heatmap(
                title=f"Tensor CUDA / HSMMLearn OMP -- {gpu_sys} vs {cpu_sys}",
                speedup_label="Tensor CUDA / HSMMLearn OMP",
                speedup_dict=speedup,
                out_path=os.path.join(OUT_ROOT, f"cuda_vs_omp_{gpu_slug}__{cpu_slug}.png"),
            )


if __name__ == "__main__":
    main()
