#!/usr/bin/env python3
"""
plot_performance_bar.py — speedup bar charts vs HSMMLearn C++.

Generates three plots per (N, T) pair:
  bars/cpu/cpu_{N}s_{T}t.png          — CPU systems only, ref = HSMMLearn C++
  bars/gpu/vscpp/gpu_vs_cpp_{N}s_{T}t.png — GPU systems, ref = slowest CPU HSMMLearn C++
  bars/gpu/vsomp/gpu_vs_omp_{N}s_{T}t.png — GPU systems, ref = fastest CPU HSMMLearn OMP

X-axis: duration D values.

Usage:
  python plot/plot_performance_bar.py              # all (N, T) pairs
  python plot/plot_performance_bar.py --states 75
  python plot/plot_performance_bar.py --states 75 --timesteps 100000
  python plot/plot_performance_bar.py --all-toolchains
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars")

# Default toolchain per system prefix
DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "gnu",
}

GPU_GENERATION = {
    "a100":         0,   # A100  SM80  ~2020
    "mi250x":       1,   # MI250X gfx90a ~2021
    "h100":         2,   # H100  SM90  ~2022
    "gh200-hopper": 3,   # H100  SM90 (GH200) ~2023
    "mi300x":       4,   # MI300X gfx942 ~2023
}

CPU_FUNCTION_ORDER = [
    "decode_tensor_viterbi_cpp",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_omp",
    "decode_tensor_viterbi_omp_opt",
]
GPU_FUNCTION_ORDER = [
    "decode_tensor_viterbi_cuda",
]
FUNCTION_LABELS = {
    "HSMMLearn_CPP":                 "HSMMLearn C++",
    "HSMMLearn_OMP":                 "HSMMLearn OMP",
    "decode_tensor_viterbi_cpp":     "Tensor C++",
    "decode_tensor_viterbi_omp":     "Tensor OMP",
    "decode_tensor_viterbi_omp_opt": "Tensor OMP-OPT",
    "decode_tensor_viterbi_cuda":    "Tensor GPU",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_systems(all_systems, use_all):
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


def _is_gpu(sys_tc, all_data):
    return any(
        "decode_tensor_viterbi_cuda" in funcs
        for funcs in all_data[sys_tc].values()
    )


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    return sorted({
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        )
        for p in glob.glob(pattern)
    })


def discover_nt_pairs():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    pairs = set()
    for p in glob.glob(pattern):
        fname = os.path.basename(p)
        parts = fname.split("_")
        try:
            n = int(parts[0].rstrip("s"))
            t = int(parts[2].rstrip("t"))
            pairs.add((n, t))
        except (IndexError, ValueError):
            pass
    return sorted(pairs)


def load_means(system, n, d, t):
    """Return {function -> {mean, std}} (warmup iteration 0 excluded)."""
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*.csv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["iteration"] != 0]
    result = {}
    for func, grp in df.groupby("function"):
        result[func] = {
            "mean": grp["elapsed_s"].mean(),
            "std":  grp["elapsed_s"].std(ddof=1),
        }
    return result


# ── Per-kind plot ─────────────────────────────────────────────────────────────

def make_plot(N, T, kind, all_systems, all_data, d_values, ref_system,
              gpu_ref="cpp"):
    """
    kind    : 'cpu' or 'gpu'
    gpu_ref : 'cpp' → ref = slowest CPU HSMMLearn C++ (conservative upper bound)
              'omp' → ref = fastest CPU HSMMLearn OMP (practical parallel baseline)
    all_data: {sys_tc: {D_val: {func: {mean, std}}}}
    Saves bars/{kind}_{N}s_{T}t.png (cpu) or
          bars/gpu_vs_{gpu_ref}_{N}s_{T}t.png (gpu); skips if no data.
    """
    if kind == "cpu":
        systems = sorted(s for s in all_systems if not _is_gpu(s, all_data))
        func_order = CPU_FUNCTION_ORDER
    else:
        systems = sorted(
            (s for s in all_systems if _is_gpu(s, all_data)),
            key=lambda s: GPU_GENERATION.get(s.split("/")[0], 99),
        )
        func_order = GPU_FUNCTION_ORDER

    # Build ordered (func, sys_tc) combos, excluding the reference HSMMLearn_CPP
    combo_set = set()
    for sys_tc in systems:
        for funcs in all_data[sys_tc].values():
            for func in funcs:
                if func != "HSMMLearn_CPP":
                    combo_set.add((func, sys_tc))

    def combo_key(c):
        func, sys_tc = c
        si = systems.index(sys_tc) if sys_tc in systems else len(systems)
        fi = func_order.index(func) if func in func_order else len(func_order)
        return (si, fi)

    ordered_combos = sorted(combo_set, key=combo_key)
    if not ordered_combos:
        return

    n_combos = len(ordered_combos)

    # Color map
    cmap_name = "tab10" if n_combos <= 10 else "tab20"
    cmap  = plt.get_cmap(cmap_name)
    n_map = 10 if cmap_name == "tab10" else 20
    colors = {combo: cmap(i / n_map) for i, combo in enumerate(ordered_combos)}

    # Reference helper.
    # CPU plots: same system's HSMMLearn C++.
    # GPU plots, gpu_ref='cpp': slowest CPU HSMMLearn C++ (conservative upper bound).
    # GPU plots, gpu_ref='omp': fastest CPU HSMMLearn OMP (practical parallel baseline).
    def get_ref(sys_tc, D_val):
        if kind == "cpu":
            d = all_data[sys_tc].get(D_val, {})
            if "HSMMLearn_CPP" in d:
                return d["HSMMLearn_CPP"]["mean"]
        cpu_sys = [s for s in all_systems if not _is_gpu(s, all_data)]
        if gpu_ref == "omp":
            candidates = [
                all_data[s].get(D_val, {}).get("HSMMLearn_OMP", {}).get("mean")
                for s in cpu_sys
            ]
            candidates = [c for c in candidates if c is not None]
            return min(candidates) if candidates else None  # fastest OMP
        # gpu_ref == "cpp" (default)
        if ref_system:
            d2 = all_data.get(ref_system, {}).get(D_val, {})
            if "HSMMLearn_CPP" in d2:
                return d2["HSMMLearn_CPP"]["mean"]
        candidates = [
            all_data[s].get(D_val, {}).get("HSMMLearn_CPP", {}).get("mean")
            for s in cpu_sys
        ]
        candidates = [c for c in candidates if c is not None]
        return max(candidates) if candidates else None  # slowest CPP

    # ── Draw ────────────────────────────────────────────────────────────────
    bar_width = 0.8 / n_combos
    x_pos     = np.arange(len(d_values), dtype=float)

    fig_w = max(8, len(d_values) * (n_combos * bar_width + 0.6) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    for ci, (func, sys_tc) in enumerate(ordered_combos):
        offset   = (ci - (n_combos - 1) / 2) * bar_width
        heights, errs = [], []
        for D_val in d_values:
            fdata = all_data[sys_tc].get(D_val, {}).get(func)
            ref   = get_ref(sys_tc, D_val)
            if fdata is None or ref is None or fdata["mean"] == 0:
                heights.append(0.0)
                errs.append(0.0)
            else:
                m = fdata["mean"]
                s = fdata["std"] or 0.0
                heights.append(ref / m)
                errs.append(ref * s / (m ** 2))
                any_data = True

        label = f"{FUNCTION_LABELS.get(func, func)} — {sys_tc}"
        ax.bar(
            x_pos + offset, heights, width=bar_width,
            color=colors[(func, sys_tc)], label=label,
            yerr=errs, capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "black"},
            zorder=2,
        )

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(1.0, color="black", lw=0.9, linestyle="--", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"D = {D}" for D in d_values], fontsize=9)
    ax.set_xlabel("Duration  D", fontsize=9)

    if kind == "gpu" and gpu_ref == "omp":
        ref_label = "HSMMLearn OMP (fastest CPU)"
        ref_note  = "ref: fastest CPU HSMMLearn OMP"
    elif kind == "gpu":
        ref_label = "HSMMLearn C++"
        ref_note  = f"ref: {ref_system}" if ref_system else "ref: slowest CPU HSMMLearn C++"
    else:
        ref_label = "HSMMLearn C++"
        ref_note  = None

    ax.set_ylabel(f"Speedup vs {ref_label}  (higher = faster)", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    if kind == "gpu":
        title = (
            f"GPU Speedup vs {ref_label}\n"
            f"N = {N} states,  T = {T:,}    ({ref_note})"
        )
    else:
        title = f"CPU Speedup vs HSMMLearn C++\nN = {N} states,  T = {T:,}"
    ax.set_title(title, fontsize=10)

    ax.legend(
        fontsize=7.5, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
        title="function — system/toolchain", title_fontsize=8,
    )

    if kind == "gpu":
        out_dir  = os.path.join(OUT_ROOT, "gpu", f"vs{gpu_ref}")
        out_path = os.path.join(out_dir, f"gpu_vs_{gpu_ref}_{N}s_{T}t.png")
    else:
        out_dir  = os.path.join(OUT_ROOT, "cpu")
        out_path = os.path.join(out_dir, f"cpu_{N}s_{T}t.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",    "-s", type=int, default=None,
                        help="Filter to this N value (default: all)")
    parser.add_argument("--timesteps", "-t", type=int, default=None,
                        help="Filter to this T value (default: all)")
    parser.add_argument("--ref-system", default=None,
                        help="System/toolchain to use as GPU reference "
                             "(default: slowest CPU HSMMLearn C++).")
    parser.add_argument("--all-toolchains", action="store_true",
                        help="Show all toolchains; by default only the "
                             "default toolchain per system is shown.")
    args = parser.parse_args()

    all_systems = _filter_systems(discover_systems(), args.all_toolchains)
    if not all_systems:
        sys.exit("Error: no results found.")

    nt_pairs = discover_nt_pairs()
    if args.states:
        nt_pairs = [(n, t) for n, t in nt_pairs if n == args.states]
    if args.timesteps:
        nt_pairs = [(n, t) for n, t in nt_pairs if t == args.timesteps]
    if not nt_pairs:
        sys.exit("Error: no (N, T) pairs match the given filters.")

    for N, T in nt_pairs:
        # Discover D values with data for this (N, T)
        d_set = set()
        for sys_tc in all_systems:
            for f in glob.glob(
                os.path.join(RESULTS_ROOT, sys_tc, f"{N}s_*_{T}t_*.csv")
            ):
                parts = os.path.basename(f).split("_")
                try:
                    d_set.add(int(parts[1].rstrip("d")))
                except (IndexError, ValueError):
                    pass
        if not d_set:
            continue
        d_values = sorted(d_set)

        # Load all data: all_data[sys_tc][D_val] = {func: {mean, std}}
        all_data = {}
        for sys_tc in all_systems:
            all_data[sys_tc] = {
                D_val: load_means(sys_tc, N, D_val, T)
                for D_val in d_values
            }

        make_plot(N, T, "cpu", all_systems, all_data, d_values, args.ref_system)
        for gpu_ref in ("cpp", "omp"):
            make_plot(N, T, "gpu", all_systems, all_data, d_values,
                      args.ref_system, gpu_ref=gpu_ref)


if __name__ == "__main__":
    main()
