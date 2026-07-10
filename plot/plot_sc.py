#!/usr/bin/env python3
"""
plot_sc.py — single-core speedup: Tens-1C vs Base-1C.

Reference: HSMMLearn C++ (Base-1C) on the same system.
Bar height: speedup = Base-1C_time / Tens-1C_time.

X-axis: N (states) as primary groups, D (duration) as sub-groups within each N.
Bars within each (N, D): one per CPU system.
One plot per T (timesteps).

Output: bars/sc/sc_{T}t.png

Usage:
  python plot/plot_sc.py                       # all T values
  python plot/plot_sc.py --timesteps 100000
  python plot/plot_sc.py --states 75           # narrow to one N
  python plot/plot_sc.py --all-toolchains
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars", "sc")

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "intel",
    "gh200-grace": "gnu14",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f"]

SYSTEM_ORDER = [
    "gh200-grace",
    "xeon8480",
    "epyc-7763",
    "epyc-7a53",
    "a64fx",
]

SYSTEM_LABELS = {
    "epyc-7763":    "AMD EPYC 7763",
    "xeon8480":     "Intel Xeon 8480+",
    "gh200-grace":  "ARM Grace",
    "epyc-7a53":    "AMD EPYC 7A53",
    "a64fx":        "A64FX",
}

SYSTEM_COLORS = {
    "epyc-7763":    "#4C72B0",
    "epyc-7a53":    "#DD8452",
    "xeon8480":     "#55A868",
    "gh200-grace":  "#C44E52",
    "a64fx":        "#8172B2",
}

_HATCH_PATTERNS = ["", "///", "...", "xxx", "|||", "---"]

BASE_FUNC   = "HSMMLearn_CPP"
TARGET_FUNC = "decode_tensor_viterbi_cpp"


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


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    return sorted({
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        )
        for p in glob.glob(pattern)
        if "metrics" not in p
    })


def discover_ndt_triples():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*.csv")
    triples = set()
    for p in glob.glob(pattern):
        if "metrics" in p:
            continue
        parts = os.path.basename(p).split("_")
        try:
            triples.add((int(parts[0].rstrip("s")),
                         int(parts[1].rstrip("d")),
                         int(parts[2].rstrip("t"))))
        except (IndexError, ValueError):
            pass
    return sorted(triples)


def load_means(system, n, d, t):
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*.csv"))
    files = [f for f in files if "metrics" not in f]
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df["iteration"].nunique() > 1:
        df = df[df["iteration"] != 0]
    result = {}
    for func, grp in df.groupby("function"):
        result[func] = {
            "mean": grp["elapsed_s"].mean(),
            "std":  grp["elapsed_s"].std(ddof=1),
        }
    return result


def _fmt_time(t):
    """Format seconds compactly: ms / s / min / h (1 decimal)."""
    if t is None:
        return ""
    if t < 1.0:
        return f"{t*1000:.0f}ms"
    if t < 60.0:
        return f"{t:.1f}s"
    if t < 3600.0:
        return f"{t/60:.1f}min"
    return f"{t/3600:.1f}h"


def make_plot(T, cpu_systems, all_data, n_values, d_values):
    n_sys = len(cpu_systems)
    if n_sys == 0:
        return

    bar_width = 0.8 / max(n_sys, 1)
    n_gap     = 0.6

    x_items = []
    pos_map = {}
    pos     = 0.0
    prev_N  = None
    for N in n_values:
        for D in d_values:
            if not any(
                all_data[s].get((N, D), {}).get(BASE_FUNC) is not None
                and all_data[s].get((N, D), {}).get(TARGET_FUNC) is not None
                for s in cpu_systems
            ):
                continue
            if prev_N is not None and N != prev_N:
                pos += n_gap
            pos_map[(N, D)] = pos
            x_items.append((N, D))
            pos += 1.0
            prev_N = N

    if not x_items:
        print(f"T={T}: no data with both {BASE_FUNC} and {TARGET_FUNC}, skipping.")
        return

    n_groups = {}
    for (N, D), p in pos_map.items():
        n_groups.setdefault(N, []).append(p)

    fig_w = max(9, len(x_items) * 0.95 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    for si, sys_tc in enumerate(cpu_systems):
        offset = (si - (n_sys - 1) / 2) * bar_width
        sname  = sys_tc.split("/")[0]
        color  = SYSTEM_COLORS.get(sname, "#999999")
        hatch  = _HATCH_PATTERNS[si % len(_HATCH_PATTERNS)]

        xpos    = []
        heights = []
        errs    = []
        for (N, D) in x_items:
            nd     = all_data[sys_tc].get((N, D), {})
            base   = nd.get(BASE_FUNC)
            target = nd.get(TARGET_FUNC)
            xpos.append(pos_map[(N, D)] + offset)
            if base is None or target is None or target["mean"] == 0:
                heights.append(0.0)
                errs.append(0.0)
            else:
                m  = target["mean"]
                s  = target["std"] or 0.0
                bm = base["mean"]
                heights.append(bm / m)
                errs.append(bm * s / (m ** 2))
                any_data = True

        ax.bar(
            xpos, heights, width=bar_width,
            color=color, hatch=hatch,
            edgecolor="black", linewidth=0.4,
            yerr=errs, capsize=2,
            error_kw={"elinewidth": 0.7, "ecolor": "black"},
            zorder=2,
        )

    if not any_data:
        plt.close(fig)
        return

    # Annotate each bar: Base-1C time, arrow, Tens-1C time — all in system color.
    # Track anchor + display-point offset to the top of the label stack on the
    # first bar, so the "Base-1C → Tens-1C" box can point exactly there.
    first_bar_xy      = None   # data coords anchor of the first bar's labels
    first_bar_top_off = None   # display-point offset (upward) to top of label stack

    for si, sys_tc in enumerate(cpu_systems):
        offset = (si - (n_sys - 1) / 2) * bar_width
        sname  = sys_tc.split("/")[0]
        color  = SYSTEM_COLORS.get(sname, "#999999")
        for (N, D) in x_items:
            nd     = all_data[sys_tc].get((N, D), {})
            base   = nd.get(BASE_FUNC)
            target = nd.get(TARGET_FUNC)
            if base is None or target is None or target["mean"] == 0:
                continue
            spd   = base["mean"] / target["mean"]
            x_bar = pos_map[(N, D)] + offset
            label_base   = _fmt_time(base["mean"])
            label_arrow  = "\u2192"
            label_target = _fmt_time(target["mean"])
            gap  = 2.0
            cpt  = 8.0   # display-points per character at fontsize 13 bold
            off0 = 2.0
            off1 = off0 + max(len(label_base),   2) * cpt + gap
            off2 = off1 + max(len(label_arrow),  1) * cpt + gap
            off3 = off2 + max(len(label_target), 2) * cpt       # top of stack
            _ann = dict(ha="center", va="bottom", fontsize=13, fontweight="bold",
                        rotation=90, clip_on=True,
                        xycoords="data", textcoords="offset points")
            ax.annotate(label_base,   xy=(x_bar, spd + 0.10), xytext=(0, off0), color=color, **_ann)
            ax.annotate(label_arrow,  xy=(x_bar, spd + 0.10), xytext=(0, off1), color="black", **_ann)
            ax.annotate(label_target, xy=(x_bar, spd + 0.10), xytext=(0, off2), color=color, **_ann)
            if first_bar_xy is None:
                first_bar_xy      = (x_bar, spd + 0.10)
                first_bar_top_off = off3

    _, ymax = ax.get_ylim()
    if any(h < 1.0 for h in heights if h > 0):
        print("\\033[1;31mError: Speedup < 1 detected!\\033[0m")
        return
    
    if T == 1000:
        ax.set_ylim(1, ymax * 1.49)
    elif T == 10000:
        ax.set_ylim(1, ymax * 1.49)
    elif T == 100000:
        ax.set_ylim(1, ymax * 1.60)
    else:
        ax.set_ylim(1, ymax * 1.60)

    # Place the "Base-1C → Tens-1C" box after set_ylim so the display
    # transform is correct.  xy is given in figure-pixel coords so the tip
    # lands at the TOP of the first bar's label stack (not at bar-top).
    if first_bar_xy is not None:
        fig.canvas.draw()
        disp_anchor  = ax.transData.transform(first_bar_xy)          # display px
        top_px       = first_bar_top_off * fig.dpi / 72.0            # points → display px
        tip_px       = (disp_anchor[0], disp_anchor[1] + top_px)
        ax.annotate(
            "Base-1C Time \u2192 Tens-1C Time",
            xy=tip_px,
            xytext=(0.02, 0.97),
            xycoords="figure pixels",
            textcoords="axes fraction",
            fontsize=9, fontweight="bold", color="black",
            ha="center", va="top",
            rotation=90,
            bbox=dict(facecolor="white", edgecolor="black",
                      boxstyle="round,pad=0.3", linewidth=0.8),
            #arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
        )

#    ax.axhline(1.0, color="black", lw=0.9, linestyle="--", zorder=3)
#    ax.annotate("Base-1C", xy=(0.99, 1.0),
#                xycoords=("axes fraction", "data"),
#                xytext=(0, 3), textcoords="offset points",
#                fontsize=12, ha="right", va="bottom", color="black",
#                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

    ax.set_xticks([pos_map[(N, D)] for (N, D) in x_items])
    ax.set_xticklabels([str(D) for (_, D) in x_items], fontsize=12)

    _blended = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    _dur_y   = -0.07   # "Duration D" label, per N-group
    _bkt_y   = -0.13   # bracket line
    _tick_h  = 0.015
    _lbl_y   = -0.155  # "N=..." label
    for N, positions in sorted(n_groups.items()):
        xl = min(positions) - 0.5
        xr = max(positions) + 0.5
        xm = (xl + xr) / 2
        ax.text(xm, _dur_y, "Duration  D",
                transform=_blended, ha="center", va="top",
                fontsize=12, clip_on=False)
        ax.plot([xl, xr], [_bkt_y, _bkt_y],
                transform=_blended, color="black", lw=0.9, clip_on=False)
        ax.plot([xl, xl], [_bkt_y, _bkt_y - _tick_h],
                transform=_blended, color="black", lw=0.9, clip_on=False)
        ax.plot([xr, xr], [_bkt_y, _bkt_y - _tick_h],
                transform=_blended, color="black", lw=0.9, clip_on=False)
        ax.text(xm, _lbl_y, f"N={N}",
                transform=_blended, ha="center", va="top",
                fontsize=12, clip_on=False)

    ax.set_xlabel("", labelpad=0)
    ax.set_ylabel("Speedup over Base-1C (higher=better)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)

    yticks = list(ax.get_yticks())
    if 1.0 not in yticks:
        yticks = sorted([1.0] + [t for t in yticks if t > 1.0])
        ax.set_yticks(yticks)
    yticklabels = [f"{int(y)}x" if y == int(y) else f"{y:.1f}x" for y in ax.get_yticks()]
    ax.set_yticklabels(yticklabels)    

    legend_handles = [
        mpatches.Patch(
            facecolor=SYSTEM_COLORS.get(sys_tc.split("/")[0], "#999999"),
            hatch=_HATCH_PATTERNS[si % len(_HATCH_PATTERNS)],
            edgecolor="black", linewidth=0.4,
            label=SYSTEM_LABELS.get(sys_tc.split("/")[0], sys_tc.split("/")[0]),
        )
        for si, sys_tc in enumerate(cpu_systems)
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="upper center",
              bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=True, framealpha=0.85)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"sc_{T}t.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",         "-s", type=int, nargs="+",
                        default=[10, 25, 75],
                        help="N values to include (default: 10 25 75). Use --states 0 for all.")
    parser.add_argument("--timesteps",      "-t", type=int, default=None)
    parser.add_argument("--all-toolchains", action="store_true")
    args = parser.parse_args()

    all_systems = _filter_systems(discover_systems(), args.all_toolchains)
    if not all_systems:
        sys.exit("Error: no results found.")

    all_triples = discover_ndt_triples()
    if args.states and args.states != [0]:
        all_triples = [(n, d, t) for n, d, t in all_triples if n in args.states]
    if args.timesteps:
        all_triples = [(n, d, t) for n, d, t in all_triples if t == args.timesteps]
    if not all_triples:
        sys.exit("Error: no data matches the given filters.")

    t_values = sorted({t for _, _, t in all_triples})

    for T in t_values:
        triples_t = [(n, d, t) for n, d, t in all_triples if t == T]
        n_values  = sorted({n for n, _, _ in triples_t})
        d_values  = sorted({d for _, d, _ in triples_t})

        all_data = {
            sys_tc: {
                (n, d): load_means(sys_tc, n, d, T)
                for n, d, _ in triples_t
            }
            for sys_tc in all_systems
        }

        cpu_systems = sorted(
            (s for s in all_systems
             if any(BASE_FUNC in all_data[s].get((n, d), {})
                    for n, d, _ in triples_t)),
            key=lambda s: SYSTEM_ORDER.index(s.split("/")[0])
                          if s.split("/")[0] in SYSTEM_ORDER else len(SYSTEM_ORDER),
        )

        if not cpu_systems:
            print(f"T={T}: no CPU systems found, skipping.")
            continue

        make_plot(T, cpu_systems, all_data, n_values, d_values)


if __name__ == "__main__":
    main()
