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
    "mi300x":    "amd",
    "epyc-7a53": "cray",   # matches epyc-7763-bigmem etc.
    "xeon8480":  "intel",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f"]

SHOW_T = {10_000, 100_000, 1_000_000}


def _filter_systems(all_systems, use_all):
    """Keep only the default toolchain per system prefix unless use_all is True."""
    kept = []
    for sys_tc in all_systems:
        system, toolchain = sys_tc.split("/", 1)
        if system in EXCLUDED_SYSTEMS:
            continue
        if use_all:
            kept.append(sys_tc)
            continue
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


_GPU_SYSTEMS = {"a100", "h100", "mi250x", "mi300x", "b200", "gh200-hopper"}

def is_gpu(sys_tc):
    system = sys_tc.split("/")[0]
    return system in _GPU_SYSTEMS


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


def _fmt_time(t):
    """Format seconds as µs / ms / s / m / h."""
    if t < 1e-3:
        return f"{t*1e6:.0f}µs"
    if t < 1.0:
        return f"{t*1000:.0f}ms"
    if t < 60.0:
        return f"{t:.1f}s"
    if t < 3600.0:
        return f"{t/60:.1f}m"
    return f"{t/3600:.1f}h"


def make_heatmap(title, speedup_label, speedup_dict, out_path, cell_labels=None, abs_times=None):
    """
    speedup_dict : {(N, D, T) -> float}  — cells with a computed speedup value.
    cell_labels  : {(N, D, T) -> str}   — override text for cells where the
                   baseline is missing (e.g. "Base\nOOM").
    Saves one figure with one panel per unique T in SHOW_T.
    """
    cell_labels = cell_labels or {}
    abs_times   = abs_times   or {}
    all_keys = set(speedup_dict) | set(cell_labels)
    if not all_keys:
        print(f"  Skipped (no data): {title}")
        return

    all_n = sorted({n for n, d, t in all_keys})
    all_d = sorted({d for n, d, t in all_keys}, reverse=True)
    all_t = sorted({t for n, d, t in all_keys if t in SHOW_T})

    if not all_t:
        print(f"  Skipped (no matching T): {title}")
        return

    fig, axes = plt.subplots(
        1, len(all_t),
        figsize=(4.5 * len(all_t), 4),
        squeeze=False,
        gridspec_kw={"wspace": 0.05},
    )
    axes = list(axes[0])

    all_vals = [v for v in speedup_dict.values() if not np.isnan(v)]
    if not all_vals:
        plt.close(fig)
        return
    vmin, vmax = min(all_vals), max(all_vals)
    _vmin = min(vmin, 0.95)
    _vmax = max(vmax, 1.05)
    norm = mcolors.TwoSlopeNorm(vmin=_vmin, vcenter=1.0, vmax=_vmax)
    cmap = "RdYlGn"

    for ax, T in zip(axes, all_t):
        matrix = np.full((len(all_d), len(all_n)), np.nan)
        for ri, D in enumerate(all_d):
            for ci, N in enumerate(all_n):
                v = speedup_dict.get((N, D, T))
                if v is not None:
                    matrix[ri, ci] = v

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

        for ri in range(len(all_d)):
            for ci in range(len(all_n)):
                key = (all_n[ci], all_d[ri], T)
                val = matrix[ri, ci]
                if key in cell_labels:
                    # Gray background + label for missing baseline
                    import matplotlib.patches as mpatches
                    ax.add_patch(mpatches.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        facecolor="#cccccc", edgecolor="none", zorder=2,
                    ))
                    abs_t = abs_times.get(key)
                    oom_label = cell_labels[key] + (f"\n({_fmt_time(abs_t)})" if abs_t else "")
                    ax.text(ci, ri, oom_label, ha="center", va="center",
                            fontsize=9, fontstyle="italic", color="#333333",
                            linespacing=1.3, zorder=3)
                elif not np.isnan(val):
                    # Pick white or black text for maximum contrast.
                    rgba  = im.cmap(im.norm(val))
                    # Relative luminance (WCAG formula)
                    r, g, b = rgba[0], rgba[1], rgba[2]
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    txt_color = "white" if lum < 0.45 else "black"
                    abs_t = abs_times.get(key)
                    label = f"{val:.1f}x\n({_fmt_time(abs_t)})" if abs_t else f"{val:.1f}x"
                    ax.text(ci, ri, label, ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_color,
                            linespacing=1.3)

        ax.set_xticks(range(len(all_n)))
        ax.set_xticklabels(all_n, fontsize=11)
        if ax is axes[0]:
            ax.set_yticks(range(len(all_d)))
            ax.set_yticklabels(all_d, fontsize=11, rotation=90, va='center')
            ax.set_ylabel("Max Duration (D)", fontsize=12)
        else:
            ax.set_yticks([])
        ax.set_xlabel("States (N)", fontsize=12)
        ax.set_title(f"T = {T:,}", fontsize=13, fontweight="bold")

    # --- Manually place colorbar so bar+label are centred together ----------
    # Reserve space at the bottom of the figure.
    fig.subplots_adjust(bottom=0.22)

    # Colorbar axes: [left, bottom, width, height] in figure fraction.
    # bar_w controls bar width; label is rendered as text to its right.
    # We'll nudge left after measuring the label, but a good starting point:
    bar_w  = 0.30   # width of the bar alone (figure fraction)
    bar_h  = 0.035  # thin bar
    bar_b  = 0.04   # bottom position (moves bar up vs subplots_adjust edge)

    # Estimate label width in figure fraction (fontsize 10 ≈ 7 px/char at 150 dpi)
    label_str = f"Speedup {speedup_label}  (x)"
    px_per_char = 7.7
    label_frac  = len(label_str) * px_per_char / (fig.get_figwidth() * 150)
    gap_frac    = 0.01   # small gap between bar and label

    total_w = bar_w + gap_frac + label_frac
    bar_l   = 0.5 - total_w / 2   # start so (bar+label) is centred

    cax = fig.add_axes([bar_l, bar_b, bar_w, bar_h])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation="horizontal",
    )
    cb.set_label("")
    cb.ax.text(
        1.0 + gap_frac / bar_w, 0.5,
        label_str,
        transform=cb.ax.transAxes,
        va="center", ha="left", fontsize=11,
    )
    # -------------------------------------------------------------------------

    os.makedirs(OUT_ROOT, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
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

    # 1. Tens-MC vs Base-MC  (one figure per CPU system)
    print("\n-- Tens-MC vs Base-MC --")
    for cpu_sys in cpu_systems:
        d = data[cpu_sys]
        speedup = {}
        oom = {}
        abs_times = {}
        for (n, dur, t, func), mean_s in d.items():
            if func == "decode_tensor_viterbi_omp_opt" and mean_s > 0:
                ref = d.get((n, dur, t, "HSMMLearn_OMP"))
                abs_times[(n, dur, t)] = mean_s
                if ref and ref > 0:
                    speedup[(n, dur, t)] = ref / mean_s
                else:
                    oom[(n, dur, t)] = "Base OOM"
        slug = cpu_sys.replace("/", "_")
        make_heatmap(
            title=f"Tens-MC vs Base-MC -- {cpu_sys}",
            speedup_label="Tens-MC / Base-MC",
            speedup_dict=speedup,
            cell_labels=oom,
            abs_times=abs_times,
            out_path=os.path.join(OUT_ROOT, f"tens-mc-vs-base-mc_{slug}.pdf"),
        )

    # 2. Tens-1C vs Base-1C  (one figure per CPU system)
    print("\n-- Tens-1C vs Base-1C --")
    for cpu_sys in cpu_systems:
        d = data[cpu_sys]
        speedup = {}
        oom = {}
        abs_times = {}
        for (n, dur, t, func), mean_s in d.items():
            if func == "decode_tensor_viterbi_cpp" and mean_s > 0:
                ref = d.get((n, dur, t, "HSMMLearn_CPP"))
                abs_times[(n, dur, t)] = mean_s
                if ref and ref > 0:
                    speedup[(n, dur, t)] = ref / mean_s
                else:
                    oom[(n, dur, t)] = "Base OOM"
        slug = cpu_sys.replace("/", "_")
        make_heatmap(
            title=f"Tens-1C vs Base-1C -- {cpu_sys}",
            speedup_label="Tens-1C / Base-1C",
            speedup_dict=speedup,
            cell_labels=oom,
            abs_times=abs_times,
            out_path=os.path.join(OUT_ROOT, f"tens-1c-vs-base-1c_{slug}.pdf"),
        )

    # 3. Tens-GPU vs Base-MC  (one figure per GPU x CPU pair)
    print("\n-- Tens-GPU vs Base-MC --")
    for gpu_sys in gpu_systems:
        for cpu_sys in cpu_systems:
            gpu_d = data[gpu_sys]
            cpu_d = data[cpu_sys]
            speedup = {}
            oom = {}
            abs_times = {}
            for (n, dur, t, func), mean_s in gpu_d.items():
                if func == "decode_tensor_viterbi_cuda" and mean_s > 0:
                    ref = cpu_d.get((n, dur, t, "HSMMLearn_OMP"))
                    abs_times[(n, dur, t)] = mean_s
                    if ref and ref > 0:
                        speedup[(n, dur, t)] = ref / mean_s
                    else:
                        oom[(n, dur, t)] = "Base OOM"
            gpu_slug = gpu_sys.replace("/", "_")
            cpu_slug = cpu_sys.replace("/", "_")
            make_heatmap(
                title=f"Tens-GPU vs Base-MC -- {gpu_sys} vs {cpu_sys}",
                speedup_label="Tens-GPU / Base-MC",
                speedup_dict=speedup,
                cell_labels=oom,
                abs_times=abs_times,
                out_path=os.path.join(OUT_ROOT, f"tens-gpu-vs-base-mc_{gpu_slug}__{cpu_slug}.pdf"),
            )


if __name__ == "__main__":
    main()
