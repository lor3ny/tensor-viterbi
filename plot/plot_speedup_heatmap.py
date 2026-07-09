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
    "h200":      "cuda",
    "mi250x":    "cray",
    "mi300x":    "amd",
    "epyc-7a53": "cray",   # matches epyc-7763-bigmem etc.
    "xeon8480":  "intel",
    "gh200-grace": "gnu14",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f", "gh200-hopper"]

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


_GPU_SYSTEMS = {"a100", "h100", "h200", "mi250x", "mi300x", "b200"}

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


def _fmt_t(t):
    """Format a timestep value as a compact string: 10000 -> '10k', 1000000 -> '1m'."""
    if t % 1_000_000 == 0:
        return f"{t // 1_000_000}m"
    if t % 1_000 == 0:
        return f"{t // 1_000}k"
    return str(t)


def make_heatmap(title, speedup_label, speedup_dict, out_path, cell_labels=None, abs_times=None, show_t=None):
    """
    speedup_dict : {(N, D, T) -> float}  — cells with a computed speedup value.
    cell_labels  : {(N, D, T) -> str}   — override text for cells where the
                   baseline is missing (e.g. "Base\nOOM").
    Saves one figure with one panel per unique T in show_t (or SHOW_T if None).
    """
    cell_labels = cell_labels or {}
    abs_times   = abs_times   or {}
    all_keys = set(speedup_dict) | set(cell_labels)
    if not all_keys:
        print(f"  Skipped (no data): {title}")
        return

    EXCLUDE_N = {100, 128}
    all_n = sorted({n for n, d, t in all_keys if n not in EXCLUDE_N})
    _show = show_t if show_t is not None else SHOW_T
    all_t = sorted({t for n, d, t in all_keys if t in _show})
    # all_d is computed per panel to allow D=10000 exclusion for T<=100K

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

    # Build a single global D list (union across all panels) so every panel
    # has the same number of rows and cells share the same height.
    _per_panel_d = {}
    for T in all_t:
        _per_panel_d[T] = {d for n, d, t in all_keys if t == T and d != 10_000}
    global_d = sorted({d for ds in _per_panel_d.values() for d in ds}, reverse=True)

    # NaN cells (excluded rows) rendered as light gray.
    import copy
    cmap_obj = copy.copy(plt.cm.get_cmap(cmap))
    cmap_obj.set_bad(color="#d0d0d0")

    for ax, T in zip(axes, all_t):
        matrix = np.full((len(global_d), len(all_n)), np.nan)
        for ri, D in enumerate(global_d):
            if D not in _per_panel_d[T]:
                continue   # leave row as NaN -> gray
            for ci, N in enumerate(all_n):
                v = speedup_dict.get((N, D, T))
                if v is not None:
                    matrix[ri, ci] = v

        im = ax.imshow(matrix, aspect="auto", cmap=cmap_obj, norm=norm)

        for ri in range(len(global_d)):
            for ci in range(len(all_n)):
                D   = global_d[ri]
                key = (all_n[ci], D, T)
                val = matrix[ri, ci]
                if D not in _per_panel_d[T]:
                    pass   # gray excluded cell, no text
                elif key in cell_labels:
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
            ax.set_yticks(range(len(global_d)))
            ax.set_yticklabels(global_d, fontsize=11, rotation=90, va='center')
            ax.set_ylabel("Max Duration (D)", fontsize=12)
        else:
            ax.set_yticks([])
        if len(all_t) > 1:
            ax.set_xlabel(f"States (N)\nT = {T:,}", fontsize=12, labelpad=4)
        else:
            ax.set_xlabel("States (N)", fontsize=12)
    # --- Colorbar at top, spanning the full heatmap width -------------------
    fig.subplots_adjust(top=0.80, bottom=0.15)

    # Derive bar bounds from the actual axes positions (figure fractions).
    pos_l = axes[0].get_position()
    pos_r = axes[-1].get_position()
    bar_l = pos_l.x0
    bar_w = pos_r.x1 - pos_l.x0
    bar_h = 0.035
    bar_b = pos_r.y1 + 0.05   # just above the heatmap

    cax = fig.add_axes([bar_l, bar_b, bar_w, bar_h])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation="horizontal",
    )
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    label_str = f"Speedup ({speedup_label.replace(' / ', ' over ')})"
    cb.set_label(label_str, fontsize=11, labelpad=8)
    cb.ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:g}x")
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
    parser.add_argument(
        "--timestep", "-t", type=int, nargs="+", default=None,
        help="Show only these T values (e.g. -t 100000 1000000). Compact suffixes used in filenames.",
    )
    args = parser.parse_args()
    show_t = set(args.timestep) if args.timestep is not None else None
    _t_suffix = "".join(f"_t{_fmt_t(t)}" for t in sorted(show_t)) if show_t is not None else ""

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
            if func == "decode_tensor_viterbi_omp" and mean_s > 0:
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
            out_path=os.path.join(OUT_ROOT, f"tens-mc-vs-base-mc_{slug}{_t_suffix}.png"),
            show_t=show_t,
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
            out_path=os.path.join(OUT_ROOT, f"tens-1c-vs-base-1c_{slug}{_t_suffix}.png"),
            show_t=show_t,
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
                out_path=os.path.join(OUT_ROOT, f"tens-gpu-vs-base-mc_{gpu_slug}__{cpu_slug}{_t_suffix}.png"),
                show_t=show_t,
            )


if __name__ == "__main__":
    main()
