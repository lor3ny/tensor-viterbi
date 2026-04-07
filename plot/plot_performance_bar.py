#!/usr/bin/env python3
"""
plot_performance_bar.py — speedup bar charts vs HSMMLearn C++.

Generates three plots per (N, T) pair:
  bars/cpu/cpu_{N}s_{T}t.png          — CPU systems only, ref = HSMMLearn C++
  bars/gpu/vshsmm/gpu_vs_hsmm_{N}s_{T}t.png    — GPU systems, ref = HSMMLearn C++ (slowest CPU)
  bars/gpu/vshsmmomp/gpu_vs_hsmmomp_{N}s_{T}t.png — GPU systems, ref = fastest CPU HSMMLearn OMP
  bars/gpu/vsopt/gpu_vs_opt_{N}s_{T}t.png       — GPU systems, ref = fastest CPU Tensor OMP-OPT

X-axis: duration D values.

Usage:
  python plot/plot_performance_bar.py              # all (N, T) pairs
  python plot/plot_performance_bar.py --states 75
  python plot/plot_performance_bar.py --states 75 --timesteps 100000
  python plot/plot_performance_bar.py --all-toolchains
"""

import argparse
import colorsys
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars")

# Default toolchain per system prefix
DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "gnu",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f"]

SYSTEM_LABELS = {
    "epyc-7763":    "AMD EPYC 7763",
    "epyc-7a53":    "AMD EPYC (7A53)",
    "xeon8480":     "Intel Xeon 8480",
    "a100":         "A100",
    "mi250x":       "MI250X",
    "h100":         "H100",
    "gh200-hopper": "GH200",
    "gh200-grace":  "ARM Grace",
    "mi300x":       "MI300X",
    "b200":         "B200",
    "a64fx":        "A64FX",
}

GPU_GENERATION = {
    "a100":         0,   # A100  SM80  ~2020
    "mi250x":       1,   # MI250X gfx90a ~2021
    "h100":         2,   # H100  SM90  ~2022
    "gh200-hopper": 3,   # H100  SM90 (GH200) ~2023
    "mi300x":       4,   # MI300X gfx942 ~2023
    "b200":         5,   # B200  SM100 ~2025
}

CPU_FUNCTION_ORDER = [
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp_opt",
]
GPU_FUNCTION_ORDER = [
    "decode_tensor_viterbi_cuda",
]
FUNCTION_LABELS = {
    "HSMMLearn_CPP":                 "HSMMLearn (Sequential)",
    "HSMMLearn_OMP":                 "HSMMLearn (OMP)",
    "decode_tensor_viterbi_cpp":     "Tensor Single-Core",
    "decode_tensor_viterbi_omp_opt": "Tensor Multi-Core",
    "decode_tensor_viterbi_cuda":    "Tensor (GPU)",
}

# Base hue per algorithm; shade+hatch vary by system within each family
FUNCTION_BASE_COLORS = {
    "HSMMLearn_OMP":                 "#4C72B0",
    "decode_tensor_viterbi_cpp":     "#DD8452",
    "decode_tensor_viterbi_omp_opt": "#C44E52",
    "decode_tensor_viterbi_cuda":    "#8172B2",
}
_HATCH_PATTERNS = ["", "///", "...", "xxx", "|||", "---"]


def _shade_color(hex_color, factor):
    """Return an RGB tuple with lightness scaled by factor (0.65=dark, 1.3=light)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lgt, sat = colorsys.rgb_to_hls(r, g, b)
    lgt = max(0.15, min(0.88, lgt * factor))
    return colorsys.hls_to_rgb(hue, lgt, sat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_systems(all_systems, use_all):
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
              gpu_ref="hsmm"):
    """
    kind    : 'cpu' or 'gpu'
    gpu_ref : 'hsmm'    → ref = slowest CPU HSMMLearn C++
              'hsmmomp' → ref = fastest CPU HSMMLearn OMP
              'opt'     → ref = fastest CPU Tensor OMP-OPT
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
                if func not in ("HSMMLearn_CPP", "decode_tensor_viterbi_omp"):
                    combo_set.add((func, sys_tc))

    def combo_key(c):
        func, sys_tc = c
        si = systems.index(sys_tc) if sys_tc in systems else len(systems)
        fi = func_order.index(func) if func in func_order else len(func_order)
        return (fi, si)

    ordered_combos = sorted(combo_set, key=combo_key)
    if not ordered_combos:
        return

    n_combos = len(ordered_combos)

    # Hatch + shade: both keyed by system index for consistency across functions
    n_sys = len(systems)
    shade_factors = [
        0.65 + (1.30 - 0.65) * i / max(n_sys - 1, 1)
        for i in range(n_sys)
    ]
    hatch_map = {
        sys_tc: _HATCH_PATTERNS[i % len(_HATCH_PATTERNS)]
        for i, sys_tc in enumerate(systems)
    }
    shade_colors = {
        (func, sys_tc): _shade_color(
            FUNCTION_BASE_COLORS.get(func, "#999999"), shade_factors[i]
        )
        for func in func_order
        for i, sys_tc in enumerate(systems)
    }

    # Reference helper.
    # CPU plots: same system's HSMMLearn C++.
    # GPU plots, gpu_ref='hsmm':    slowest CPU HSMMLearn C++.
    # GPU plots, gpu_ref='hsmmomp': fastest CPU HSMMLearn OMP.
    # GPU plots, gpu_ref='opt':     fastest CPU Tensor OMP-OPT (best CPU parallel baseline).
    def get_ref(sys_tc, D_val):
        if kind == "cpu":
            d = all_data[sys_tc].get(D_val, {})
            if "HSMMLearn_CPP" in d:
                return d["HSMMLearn_CPP"]["mean"]
        cpu_sys = [s for s in all_systems if not _is_gpu(s, all_data)]
        if gpu_ref == "hsmmomp":
            candidates = [
                all_data[s].get(D_val, {}).get("HSMMLearn_OMP", {}).get("mean")
                for s in cpu_sys
            ]
            candidates = [c for c in candidates if c is not None]
            return min(candidates) if candidates else None  # fastest HSMMLearn OMP
        if gpu_ref == "opt":
            candidates = [
                all_data[s].get(D_val, {}).get("decode_tensor_viterbi_omp_opt", {}).get("mean")
                for s in cpu_sys
            ]
            candidates = [c for c in candidates if c is not None]
            return min(candidates) if candidates else None  # fastest Tensor OMP-OPT
        # gpu_ref == "hsmm": slowest CPU HSMMLearn C++
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

    # ── Group combos by function for bracket annotations ─────────────────────
    from itertools import groupby as _groupby
    func_groups = []
    for _func, _it in _groupby(ordered_combos, key=lambda c: c[0]):
        func_groups.append((_func, [c[1] for c in _it]))

    _n_gaps   = max(len(func_groups) - 1, 0)
    group_gap = 0.04 if _n_gaps > 0 else 0.0
    bar_width = (0.8 - _n_gaps * group_gap) / max(n_combos, 1)
    combo_offsets = {}
    _pos = -0.4 + bar_width / 2
    for _func, _syss in func_groups:
        for _sys_tc in _syss:
            combo_offsets[(_func, _sys_tc)] = _pos
            _pos += bar_width
        _pos += group_gap

    # ── Draw ────────────────────────────────────────────────────────────────
    x_pos = np.arange(len(d_values), dtype=float)

    fig_w = max(8, len(d_values) * 1.6 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    for func, sys_tc in ordered_combos:
        offset = combo_offsets[(func, sys_tc)]
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

        ax.bar(
            x_pos + offset, heights, width=bar_width,
            color=shade_colors.get((func, sys_tc), (0.6, 0.6, 0.6)),
            hatch=hatch_map.get(sys_tc, ""),
            edgecolor="black", linewidth=0.4,
            yerr=errs, capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "black"},
            zorder=2,
        )

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(1.0, color="black", lw=0.9, linestyle="--", zorder=3)
    ax.annotate("baseline", xy=(0.99, 1.0),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points",
                fontsize=9, ha="right", va="bottom", color="black",
                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(D) for D in d_values], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # ── Algorithm group brackets below each D tick ────────────────────────────
    if len(func_groups) > 1:
        import matplotlib.transforms as _mtrans
        _blended = _mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        _bkt_y  = -0.08
        _tick_h = 0.015
        _lbl_y  = -0.10
        _short  = {
            "HSMMLearn_OMP":                 "Baseline\n(M.-Core)",
            "decode_tensor_viterbi_cpp":     "Tensor\n(S.-Core)",
            "decode_tensor_viterbi_omp_opt": "Tensor\n(M.-Core)",
            "decode_tensor_viterbi_cuda":    "Tensor\n(GPU)",
        }
        for _xd in x_pos:
            for _gfunc, _gsyss in func_groups:
                _xl = _xd + combo_offsets[(_gfunc, _gsyss[0])]  - bar_width / 2
                _xr = _xd + combo_offsets[(_gfunc, _gsyss[-1])] + bar_width / 2
                _xm = (_xl + _xr) / 2
                ax.plot([_xl, _xr], [_bkt_y, _bkt_y],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.plot([_xl, _xl], [_bkt_y, _bkt_y - _tick_h],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.plot([_xr, _xr], [_bkt_y, _bkt_y - _tick_h],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.text(_xm, _lbl_y, _short.get(_gfunc, _gfunc),
                        transform=_blended, ha="center", va="top",
                        fontsize=7.5, clip_on=False, linespacing=1.3)
    ax.set_xlabel("Duration  D", fontsize=11, labelpad=35 if len(func_groups) > 1 else 8)

    if kind == "gpu" and gpu_ref == "hsmmomp":
        ref_label = "HSMMLearn OMP (fastest CPU)"
        ref_note  = "ref: fastest CPU HSMMLearn OMP"
    elif kind == "gpu" and gpu_ref == "opt":
        ref_label = "Tensor OMP-OPT (fastest CPU)"
        ref_note  = "ref: fastest CPU Tensor OMP-OPT"
    elif kind == "gpu":
        ref_label = "HSMMLearn C++"  # hsmm
        ref_note  = f"ref: {ref_system}" if ref_system else "ref: slowest CPU HSMMLearn C++"
    else:
        ref_label = "HSMMLearn C++"
        ref_note  = None

    ax.set_ylabel("Speedup vs Single Core Baseline  (higher = faster)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend: one entry per unique system (dedup), hatch on neutral background
    _seen_sys = set()
    legend_handles = []
    for sys_tc in systems:
        if sys_tc in _seen_sys:
            continue
        _seen_sys.add(sys_tc)
        _sname = SYSTEM_LABELS.get(sys_tc.split('/')[0], sys_tc.split('/')[0])
        legend_handles.append(mpatches.Patch(
            facecolor=(0.88, 0.88, 0.88),
            hatch=hatch_map.get(sys_tc, ""),
            edgecolor="black", linewidth=0.4,
            label=_sname,
        ))
    ax.legend(
        handles=legend_handles,
        fontsize=9, loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=1,
        frameon=True,
        framealpha=0.85,
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
        for gpu_ref in ("hsmm", "hsmmomp", "opt"):
            make_plot(N, T, "gpu", all_systems, all_data, d_values,
                      args.ref_system, gpu_ref=gpu_ref)


if __name__ == "__main__":
    main()
