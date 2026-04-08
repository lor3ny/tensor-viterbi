#!/usr/bin/env python3
"""
plot_mc_gpu.py — multicore + GPU speedup: Base-MC, Tens-MC, Tens-GPU.

Reference:
  CPU systems: each system's own Base-MC (HSMMLearn OMP). Base-MC bars = 1.0.
               Tens-MC height = own Base-MC_time / own Tens-MC_time.
  GPU systems: best (fastest) CPU Base-MC time across all CPU systems.
               Tens-GPU height = best_cpu_base_mc_time / GPU_time.

X-axis: D values. Groups per D: algorithm order with bracket annotations.
One plot per (N, T).

Output: bars/mc_gpu/mc_gpu_{N}s_{T}t.png

Usage:
  python plot/plot_mc_gpu.py
  python plot/plot_mc_gpu.py --states 75 --timesteps 100000
  python plot/plot_mc_gpu.py --all-toolchains
"""

import argparse
import colorsys
import glob
import os
import sys
from itertools import groupby as _groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars", "mc_gpu")

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "intel",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f"]

SYSTEM_LABELS = {
    "epyc-7763":    "AMD EPYC 7763",
    "epyc-7a53":    "AMD EPYC 7A53",
    "xeon8480":     "Intel Xeon 8480",
    "a100":         "A100",
    "mi250x":       "MI250X",
    "h100":         "H100",
    "gh200-hopper": "H200",
    "gh200-grace":  "ARM Grace",
    "mi300x":       "MI300X",
    "b200":         "B200",
    "a64fx":        "A64FX",
}

# CPU display order and per-system colors (matches plot_sc.py)
CPU_ORDER = ["gh200-grace", "xeon8480", "epyc-7763", "epyc-7a53", "a64fx"]
CPU_COLORS = {
    "epyc-7763":   "#4C72B0",
    "epyc-7a53":   "#DD8452",
    "xeon8480":    "#55A868",
    "gh200-grace": "#C44E52",
    "a64fx":       "#8172B2",
}

# GPU display order and distinct per-system colors
GPU_GENERATION = {
    "a100":         0,
    "h100":         1,
    "gh200-hopper": 2,
    "b200":         3,
    "mi250x":       4,
    "mi300x":       5,
}
GPU_COLORS = {
    "a100":         "#4393C3",
    "h100":         "#74C476",
    "gh200-hopper": "#FD8D3C",
    "b200":         "#D6604D",
    "mi250x":       "#9E9AC8",
    "mi300x":       "#E7298A",
}

FUNC_ORDER = [
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_omp_opt",
    "decode_tensor_viterbi_cuda",
]
FUNC_LABELS = {
    "HSMMLearn_OMP":                 "Base-MC",
    "decode_tensor_viterbi_omp_opt": "Tens-MC",
    "decode_tensor_viterbi_cuda":    "Tens-GPU",
}
FUNC_BASE_COLORS = {
    "HSMMLearn_OMP":                 "#4C72B0",
    "decode_tensor_viterbi_omp_opt": "#C44E52",
    "decode_tensor_viterbi_cuda":    "#8172B2",
}

_HATCH_PATTERNS = ["", "///", "...", "xxx", "|||", "---"]
BASE_MC_FUNC = "HSMMLearn_OMP"


def _shade_color(hex_color, factor):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    hue, lgt, sat = colorsys.rgb_to_hls(r, g, b)
    lgt = max(0.15, min(0.88, lgt * factor))
    return colorsys.hls_to_rgb(hue, lgt, sat)


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
    return any("decode_tensor_viterbi_cuda" in funcs
               for funcs in all_data[sys_tc].values())


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    return sorted({
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        )
        for p in glob.glob(pattern) if "metrics" not in p
    })


def discover_nt_pairs():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    pairs = set()
    for p in glob.glob(pattern):
        if "metrics" in p:
            continue
        parts = os.path.basename(p).split("_")
        try:
            pairs.add((int(parts[0].rstrip("s")), int(parts[2].rstrip("t"))))
        except (IndexError, ValueError):
            pass
    return sorted(pairs)


def load_means(system, n, d, t):
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*.csv"))
    files = [f for f in files if "metrics" not in f]
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["iteration"] != 0]
    result = {}
    for func, grp in df.groupby("function"):
        result[func] = {"mean": grp["elapsed_s"].mean(),
                        "std":  grp["elapsed_s"].std(ddof=1)}
    return result


def _fmt_time(t):
    """Format seconds compactly: ms / s / min / h."""
    if t is None:
        return ""
    if t < 1.0:
        return f"{t*1000:.0f}ms"
    if t < 60.0:
        return f"{t:.1f}s"
    if t < 3600.0:
        return f"{t/60:.1f}min"
    return f"{t/3600:.1f}h"


def make_plot(N, T, all_systems, all_data, d_values):
    cpu_systems = sorted(s for s in all_systems if not _is_gpu(s, all_data))
    gpu_systems = sorted(
        (s for s in all_systems if _is_gpu(s, all_data)),
        key=lambda s: GPU_GENERATION.get(s.split("/")[0], 99),
    )

    # Only Tens-MC and Tens-GPU bars (skip Base-MC)
    ordered_combos = []
    for func in FUNC_ORDER:
        if func == BASE_MC_FUNC:
            continue
        sys_list = cpu_systems if func != "decode_tensor_viterbi_cuda" else gpu_systems
        for sys_tc in sys_list:
            if any(func in all_data[sys_tc].get(D, {}) for D in d_values):
                ordered_combos.append((func, sys_tc))

    if not ordered_combos:
        return

    # Sort CPU systems by CPU_ORDER; GPU systems already sorted by GPU_GENERATION
    cpu_systems = sorted(
        cpu_systems,
        key=lambda s: CPU_ORDER.index(s.split("/")[0])
                      if s.split("/")[0] in CPU_ORDER else len(CPU_ORDER),
    )

    # Per-system colors (no shading — each system has its own distinct color)
    sys_colors = {}
    for sys_tc in cpu_systems:
        sname = sys_tc.split("/")[0]
        sys_colors[sys_tc] = CPU_COLORS.get(sname, "#999999")
    for sys_tc in gpu_systems:
        sname = sys_tc.split("/")[0]
        sys_colors[sys_tc] = GPU_COLORS.get(sname, "#aaaaaa")

    hatch_map = {}
    for si, sys_tc in enumerate(cpu_systems):
        hatch_map[sys_tc] = _HATCH_PATTERNS[si % len(_HATCH_PATTERNS)]
    for si, sys_tc in enumerate(gpu_systems):
        hatch_map[sys_tc] = _HATCH_PATTERNS[si % len(_HATCH_PATTERNS)]

    func_groups = []
    for func, it in _groupby(ordered_combos, key=lambda c: c[0]):
        func_groups.append((func, [c[1] for c in it]))

    n_combos  = len(ordered_combos)
    n_gaps    = max(len(func_groups) - 1, 0)
    group_gap = 0.04 if n_gaps > 0 else 0.0
    bar_width = (0.8 - n_gaps * group_gap) / max(n_combos, 1)

    combo_offsets = {}
    _pos = -0.4 + bar_width / 2
    for func, syss in func_groups:
        for sys_tc in syss:
            combo_offsets[(func, sys_tc)] = _pos
            _pos += bar_width
        _pos += group_gap

    def best_cpu_base_mc(D_val):
        times = [all_data[s].get(D_val, {}).get(BASE_MC_FUNC, {}).get("mean")
                 for s in cpu_systems]
        times = [t for t in times if t is not None]
        return min(times) if times else None

    def get_ref_time(func, sys_tc, D_val):
        """Return (ref_mean, tens_mean, err) or (None, None, None)."""
        fdata = all_data[sys_tc].get(D_val, {}).get(func)
        if fdata is None or fdata["mean"] == 0:
            return None, None, None
        m = fdata["mean"]
        s = fdata["std"] or 0.0
        if func != "decode_tensor_viterbi_cuda":
            ref_data = all_data[sys_tc].get(D_val, {}).get(BASE_MC_FUNC)
            if ref_data is None:
                return None, None, None
            ref = ref_data["mean"]
        else:
            ref = best_cpu_base_mc(D_val)
            if ref is None:
                return None, None, None
        spd = ref / m
        err = ref * s / (m ** 2)
        return ref, m, err

    x_pos = np.arange(len(d_values), dtype=float)
    fig_w = max(8, len(d_values) * 1.6 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    bar_data = {}   # (func, sys_tc, D_idx) -> (ref, m, spd, err, x_bar)
    for func, sys_tc in ordered_combos:
        offset = combo_offsets[(func, sys_tc)]
        heights = []
        errs    = []
        for di, D_val in enumerate(d_values):
            ref, m, err = get_ref_time(func, sys_tc, D_val)
            if ref is None:
                heights.append(0.0); errs.append(0.0)
            else:
                spd = ref / m
                heights.append(spd); errs.append(err); any_data = True
                bar_data[(func, sys_tc, di)] = (ref, m, spd, err, x_pos[di] + offset)
        ax.bar(
            x_pos + offset, heights, width=bar_width,
            color=sys_colors.get(sys_tc, "#999999"),
            hatch=hatch_map.get(sys_tc, ""),
            edgecolor="black", linewidth=0.4,
            yerr=errs, capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "black"},
            zorder=2,
        )

    if not any_data:
        plt.close(fig); return

    # --- bar top annotations: base_time → tens_time (vertical, rotated 90°) ---
    _ann = dict(ha="center", va="bottom", fontsize=9, fontweight="bold",
                rotation=90, clip_on=True,
                xycoords="data", textcoords="offset points")
    gap = 2.0
    cpt = 6.2   # approx display-points per character at fontsize 9 bold
    first_bar_xy      = None
    first_bar_top_off = None
    for (func, sys_tc, di), (ref, m, spd, err, x_bar) in sorted(bar_data.items()):
        color = sys_colors.get(sys_tc, "#999999")
        lbl_base = _fmt_time(ref)
        lbl_arr  = "→"
        lbl_tens = _fmt_time(m)
        off0 = 2.0
        off1 = off0 + max(len(lbl_base), 2) * cpt + gap
        off2 = off1 + max(len(lbl_arr),  1) * cpt + gap
        off3 = off2 + max(len(lbl_tens), 2) * cpt
        ax.annotate(lbl_base, xy=(x_bar, spd + 0.10), xytext=(0, off0), color=color, **_ann)
        ax.annotate(lbl_arr,  xy=(x_bar, spd + 0.10), xytext=(0, off1), color="black", **_ann)
        ax.annotate(lbl_tens, xy=(x_bar, spd + 0.10), xytext=(0, off2), color=color, **_ann)
        if first_bar_xy is None:
            first_bar_xy      = (x_bar, spd + 0.10)
            first_bar_top_off = off3

    # --- sub-1 annotations: red label at y=1 for any speedup < 1 ---
    for (func, sys_tc, di), (ref, m, spd, err, x_bar) in sorted(bar_data.items()):
        if spd < 1.0:
            ax.annotate(
                f"▼{spd:.2f}x",
                xy=(x_bar, 1.0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#cc0000", clip_on=False, zorder=5,
            )

    # --- y-axis: start at 1, tick labels as "Nx" ---
    _, ymax = ax.get_ylim()
    ax.set_ylim(1.0, ymax * 1.30)
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}x"))
    ax.tick_params(axis="y", labelsize=10)

    # Place "Base-MC Time → Tens-MC/GPU Time" box (top-right)
    if first_bar_xy is not None:
        fig.canvas.draw()
        disp_anchor = ax.transData.transform(first_bar_xy)
        top_px      = first_bar_top_off * fig.dpi / 72.0
        tip_px      = (disp_anchor[0], disp_anchor[1] + top_px)
        ax.annotate(
            "Base-MC Time → Tens-MC/GPU Time",
            xy=tip_px,
            xytext=(0.98, 0.97),
            xycoords="figure pixels",
            textcoords="axes fraction",
            fontsize=9, fontweight="bold", color="black",
            ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="black",
                      boxstyle="round,pad=0.3", linewidth=0.8),
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(D) for D in d_values], fontsize=10)

    if len(func_groups) > 1:
        _blended = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        _bkt_y = -0.08; _tick_h = 0.015; _lbl_y = -0.105
        for _xd in x_pos:
            for gfunc, gsyss in func_groups:
                xl = _xd + combo_offsets[(gfunc, gsyss[0])]  - bar_width / 2
                xr = _xd + combo_offsets[(gfunc, gsyss[-1])] + bar_width / 2
                xm = (xl + xr) / 2
                ax.plot([xl, xr], [_bkt_y, _bkt_y],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.plot([xl, xl], [_bkt_y, _bkt_y - _tick_h],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.plot([xr, xr], [_bkt_y, _bkt_y - _tick_h],
                        transform=_blended, color="black", lw=0.9, clip_on=False)
                ax.text(xm, _lbl_y, FUNC_LABELS.get(gfunc, gfunc),
                        transform=_blended, ha="center", va="top",
                        fontsize=7.5, clip_on=False)

    labelpad = 35 if len(func_groups) > 1 else 8
    ax.set_xlabel("Duration  D", fontsize=11, labelpad=labelpad)
    ax.set_ylabel("Speedup vs Base-MC  (higher = faster)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # --- legend: CPUs then GPUs with separator ---
    # Build two-column legend: left = CPUs, right = GPUs.
    # Interleave [cpu_header, gpu_header, cpu0, gpu0, cpu1, gpu1, ...] so
    # ncol=2 fills left column with CPUs and right column with GPUs.
    _blank = mpatches.Patch(visible=False, label="")
    _cpu_handles = []
    if cpu_systems:
        _cpu_handles.append(mpatches.Patch(visible=False, label="CPUs"))
        _seen = set()
        for sys_tc in cpu_systems:
            if sys_tc in _seen: continue
            _seen.add(sys_tc)
            sname = sys_tc.split("/")[0]
            _cpu_handles.append(mpatches.Patch(
                facecolor=sys_colors.get(sys_tc, "#999999"),
                hatch=hatch_map.get(sys_tc, ""),
                edgecolor="black", linewidth=0.4,
                label=SYSTEM_LABELS.get(sname, sname),
            ))
    _gpu_handles = []
    if gpu_systems:
        _gpu_handles.append(mpatches.Patch(visible=False, label="GPUs"))
        _seen = set()
        for sys_tc in gpu_systems:
            if sys_tc in _seen: continue
            _seen.add(sys_tc)
            sname = sys_tc.split("/")[0]
            _gpu_handles.append(mpatches.Patch(
                facecolor=sys_colors.get(sys_tc, "#aaaaaa"),
                hatch=hatch_map.get(sys_tc, ""),
                edgecolor="black", linewidth=0.4,
                label=SYSTEM_LABELS.get(sname, sname),
            ))
    # Matplotlib fills legend column-by-column with ncol=2.
    # Pad both lists to equal length so each fills exactly one column.
    n_rows = max(len(_cpu_handles), len(_gpu_handles))
    _cpu_handles += [_blank] * (n_rows - len(_cpu_handles))
    _gpu_handles += [_blank] * (n_rows - len(_gpu_handles))
    legend_handles = _cpu_handles + _gpu_handles
    ncols = 2 if (_cpu_handles and _gpu_handles) else 1

    ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), ncol=ncols, frameon=True, framealpha=0.85,
              handlelength=1.5, handletextpad=0.5, columnspacing=1.0)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"mc_gpu_{N}s_{T}t.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",         "-s", type=int, default=None)
    parser.add_argument("--timesteps",      "-t", type=int, default=None)
    parser.add_argument("--all-toolchains", action="store_true")
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
        d_set = set()
        for sys_tc in all_systems:
            for f in glob.glob(
                os.path.join(RESULTS_ROOT, sys_tc, f"{N}s_*_{T}t_*.csv")
            ):
                if "metrics" in f:
                    continue
                parts = os.path.basename(f).split("_")
                try:
                    d_set.add(int(parts[1].rstrip("d")))
                except (IndexError, ValueError):
                    pass
        if not d_set:
            continue
        d_values = sorted(d_set)

        all_data = {
            sys_tc: {D: load_means(sys_tc, N, D, T) for D in d_values}
            for sys_tc in all_systems
        }

        make_plot(N, T, all_systems, all_data, d_values)


if __name__ == "__main__":
    main()
