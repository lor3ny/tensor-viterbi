#!/usr/bin/env python3
"""
plot_speedup.py — per-timestep speedup heatmaps.

Usage:
  # CPU comparisons (cpp and omp):
  python plot/plot_speedup.py --cpu-system epyc-7763-bigmem

  # All three (including CUDA vs OMP baseline from one or more CPU systems):
  python plot/plot_speedup.py --cpu-system epyc-7763-bigmem --cpu-system xeon8480 --gpu-system mi250x

For each unique T value found in the data this script produces one PNG per
comparison type:
  <out_dir>/<cpu>/<T>t_cpp_speedup.png               — HSMMLearn_CPP / decode_tensor_viterbi_cpp
  <out_dir>/<cpu>/<T>t_omp_speedup.png               — HSMMLearn_OMP / decode_tensor_viterbi_omp
  <out_dir>/<gpu>/<T>t_cuda_speedup_vs_<cpu>.png     — HSMMLearn_OMP (cpu) / decode_tensor_viterbi_cuda (gpu)

Speedup > 1 means the tensor version is faster than the baseline.
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


# ── helpers ───────────────────────────────────────────────────────────────────

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.dirname(os.path.abspath(__file__))  # plot/


def load_system(system: str) -> pd.DataFrame:
    """Load all CSVs for a system and return mean elapsed per (function,N,D,T)."""
    pattern = os.path.join(RESULTS_ROOT, system, "*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: no CSV files found for system '{system}' ({pattern})", file=sys.stderr)
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return (
        df.groupby(["function", "n_states", "timesteps", "max_duration"], as_index=False)
        ["elapsed_s"].mean()
    )


def pivot(agg: pd.DataFrame, func: str, states: list, durs: list) -> np.ndarray:
    """Build a (len(durs) x len(states)) matrix of mean elapsed_s for the given function."""
    sub = agg[agg["function"] == func]
    mat = np.full((len(durs), len(states)), np.nan)
    for _, row in sub.iterrows():
        if row["n_states"] in states and row["max_duration"] in durs:
            ri = durs.index(row["max_duration"])
            ci = states.index(row["n_states"])
            mat[ri, ci] = row["elapsed_s"]
    return mat


def save_heatmap(speedup_mat: np.ndarray,
                 states: list, durs: list,
                 T: int, out_path: str,
                 title: str, cbar_label: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    valid = speedup_mat[~np.isnan(speedup_mat)]
    if valid.size == 0:
        print(f"  skipping {os.path.basename(out_path)} — no data")
        return

    # Diverging colormap centred at 1.0; red < 1 (tensor slower), green > 1 (tensor faster)
    max_dev = max(abs(valid.max() - 1.0), abs(1.0 - valid.min()), 0.5)
    norm = mcolors.TwoSlopeNorm(vcenter=1.0,
                                vmin=1.0 - max_dev,
                                vmax=1.0 + max_dev)
    cmap = "RdYlGn"

    fig, ax = plt.subplots(figsize=(max(5, len(states) * 0.9),
                                    max(4, len(durs) * 0.85)))

    im = ax.imshow(speedup_mat, aspect="auto", cmap=cmap, norm=norm)

    for ri in range(len(durs)):
        for ci in range(len(states)):
            val = speedup_mat[ri, ci]
            if not np.isnan(val):
                # dark text on light cells, white text on dark cells
                normed = norm(val)
                text_color = "white" if normed < 0.25 or normed > 0.80 else "black"
                ax.text(ci, ri, f"{val:.2f}x",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=10)
    ax.set_yticks(range(len(durs)))
    ax.set_yticklabels(durs, fontsize=10)
    ax.set_xlabel("States (N)", fontsize=11)
    ax.set_ylabel("Max Duration (D)", fontsize=11)
    ax.set_title(f"{title}\nT = {T:,} timesteps", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-system", required=True, action="append", dest="cpu_systems",
                        metavar="CPU_SYSTEM",
                        help="System name for CPU results; may be repeated (e.g. epyc-7763-bigmem)")
    parser.add_argument("--gpu-system", default=None,
                        help="System name for GPU/CUDA results (e.g. mi250x)")
    args = parser.parse_args()

    cpu_systems = args.cpu_systems  # list of one or more CPU system names
    gpu_agg     = load_system(args.gpu_system) if args.gpu_system else pd.DataFrame()

    # Load and validate all CPU systems
    cpu_data: dict[str, pd.DataFrame] = {}
    for cpu_sys in cpu_systems:
        agg = load_system(cpu_sys)
        if not agg.empty:
            cpu_data[cpu_sys] = agg

    if not cpu_data and gpu_agg.empty:
        print("Error: no data found.", file=sys.stderr)
        sys.exit(1)

    # Derive axis values from the first available CPU system (or GPU if no CPU)
    ref    = next(iter(cpu_data.values())) if cpu_data else gpu_agg
    states = sorted(ref["n_states"].unique().tolist())
    durs   = sorted(ref["max_duration"].unique().tolist(), reverse=True)  # high -> low on y-axis

    out_gpu = os.path.join(OUT_ROOT, args.gpu_system) if args.gpu_system else None

    # ── CPU heatmaps (one set of PNGs per CPU system) ────────────────────────
    comparisons_cpu = [
        # (numerator_func, denominator_func, filename_suffix, plot_title, cbar_label)
        ("HSMMLearn_CPP", "decode_tensor_viterbi_cpp",
         "cpp_speedup",
         "Speedup: Tensor C++ vs HSMMLearn C++",
         "HSMMLearn_CPP / Tensor C++  (×)"),
        ("HSMMLearn_OMP", "decode_tensor_viterbi_omp",
         "omp_speedup",
         "Speedup: Tensor OMP vs HSMMLearn OMP",
         "HSMMLearn_OMP / Tensor OMP  (×)"),
    ]

    for cpu_sys, cpu_agg in cpu_data.items():
        out_cpu = os.path.join(OUT_ROOT, cpu_sys)
        T_vals  = sorted(cpu_agg["timesteps"].unique().tolist())
        for T in T_vals:
            sub = cpu_agg[cpu_agg["timesteps"] == T]
            for num_func, den_func, suffix, title, cbar_label in comparisons_cpu:
                num_mat = pivot(sub, num_func, states, durs)
                den_mat = pivot(sub, den_func, states, durs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    speedup = np.where(den_mat > 0, num_mat / den_mat, np.nan)
                out_path = os.path.join(out_cpu, f"{T}t_{suffix}.png")
                save_heatmap(speedup, states, durs, T, out_path, title, cbar_label)

    # ── CUDA heatmaps (one set per GPU×CPU combination) ──────────────────────
    if not gpu_agg.empty and cpu_data:
        cuda_T_vals = sorted(gpu_agg["timesteps"].unique().tolist())
        for cpu_sys, cpu_agg in cpu_data.items():
            for T in cuda_T_vals:
                gpu_sub      = gpu_agg[gpu_agg["timesteps"] == T]
                cpu_sub      = cpu_agg[cpu_agg["timesteps"] == T]
                cuda_mat     = pivot(gpu_sub, "decode_tensor_viterbi_cuda", states, durs)
                omp_base_mat = pivot(cpu_sub, "HSMMLearn_OMP",              states, durs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    speedup = np.where(cuda_mat > 0, omp_base_mat / cuda_mat, np.nan)
                out_path = os.path.join(out_gpu, f"{T}t_cuda_speedup_vs_{cpu_sys}.png")
                save_heatmap(speedup, states, durs, T, out_path,
                             f"Speedup: Tensor CUDA vs HSMMLearn OMP\n"
                             f"(GPU: {args.gpu_system} / CPU baseline: {cpu_sys})",
                             "HSMMLearn_OMP (CPU) / Tensor CUDA (GPU)  (×)")


if __name__ == "__main__":
    main()
