#!/usr/bin/env python3
"""
plot_energy.py — stacked energy (J/iter) and average power (W) bar charts.

Generates both energy and power plots by default (use --metric to restrict).

For each (N, T):
  bars/energy/cpu/{cpu}_{N}s_{T}t_energy.png
  bars/energy/gpu/{gpu}_{N}s_{T}t_energy.png
  bars/power/cpu/{cpu}_{N}s_{T}t_power.png
  bars/power/gpu/{gpu}_{N}s_{T}t_power.png

Stacked components: cpu_energy, memory_energy, accel0..3_energy, others.

Usage:
  python plot/plot_energy.py
  python plot/plot_energy.py --states 75 --timesteps 100000
  python plot/plot_energy.py --metric power
  python plot/plot_energy.py --all-toolchains
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars")

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "h200":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "intel",
    "gh200-grace": "gnu14",
}

EXCLUDED_SYSTEMS = ["epyc-7763-bigmem", "epyc-9474f", "gh200-hopper"]

SYSTEM_LABELS = {
    "epyc-7763":    "AMD EPYC 7763",
    "epyc-7a53":    "AMD EPYC 7A53",
    "xeon8480":     "Intel Xeon 8480+",
    "a100":         "A100",
    "mi250x":       "MI250X",
    "h100":         "H100",
    "h200":          "H200",
    "gh200-hopper": "H100",
    "gh200-grace":  "ARM Grace",
    "mi300x":       "MI300X",
    "b200":         "B200",
    "a64fx":        "A64FX",
}

GPU_GENERATION = {
    "a100":         0,
    "mi250x":       1,
    "h100":         2,
    "h200":          3,
    "mi300x":       4,
    "b200":         5,
}

CPU_FUNCTION_ORDER = [
    "HSMMLearn_CPP",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp_opt",
]
GPU_FUNCTION_ORDER = ["decode_tensor_viterbi_cuda"]
FUNCTION_LABELS = {
    "HSMMLearn_CPP":                 "Base-1C",
    "HSMMLearn_OMP":                 "Base-MC",
    "decode_tensor_viterbi_cpp":     "Tens-1C",
    "decode_tensor_viterbi_omp_opt": "Tens-MC",
    "decode_tensor_viterbi_cuda":    "Tens-GPU",
}

ACCEL_NAMES = [f"accel{i}_energy" for i in range(4)]
ORDERED_COMPONENTS = ["cpu_energy", "memory_energy"] + ACCEL_NAMES + ["others"]

COMPONENT_COLORS = {
    "cpu_energy":    "#4e79a7",
    "memory_energy": "#59a14f",
    "accel0_energy": "#f28e2b",
    "accel1_energy": "#e15759",
    "accel2_energy": "#b07aa1",
    "accel3_energy": "#ff9da7",
    "others":        "#bab0ac",
}
COMPONENT_LABELS = {
    "cpu_energy":    "CPU",
    "memory_energy": "Memory",
    "accel0_energy": "GPU",
    "accel1_energy": "Accel 1",
    "accel2_energy": "Accel 2",
    "accel3_energy": "Accel 3",
    "others":        "Others",
}

HATCHES = ["", "///", "\\\\", "xxx", "...", "|||", "---"]

# ── Combined EPYC7A53 + MI250X plot ─────────────────────────────────────────
# Accel channels to *exclude* per system (removed from total and breakdown).
SYSTEM_ACCEL_EXCLUDE = {
    "mi250x":    {"accel1_energy", "accel2_energy", "accel3_energy"},
    "epyc-7a53": {"accel0_energy", "accel1_energy", "accel2_energy", "accel3_energy"},
}
COMBINED_FUNC_ORDER = [
    "HSMMLearn_CPP",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp_opt",
    "decode_tensor_viterbi_cuda",
]
COMBINED_SYSTEM_COLORS = {
    "epyc-7a53": "#DD8452",
    "mi250x":    "#9E9AC8",
}


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


def _has_gpu_func(sys_tc_data):
    for d_data in sys_tc_data.values():
        if any("cuda" in func for func in d_data):
            return True
    return False


def discover_systems():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*_metrics.csv")
    return sorted({
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(p))),
            os.path.basename(os.path.dirname(p)),
        )
        for p in glob.glob(pattern)
    })


def discover_nt_pairs():
    pattern = os.path.join(RESULTS_ROOT, "*", "*", "*_metrics.csv")
    pairs = set()
    for p in glob.glob(pattern):
        parts = os.path.basename(p).split("_")
        try:
            pairs.add((int(parts[0].rstrip("s")), int(parts[2].rstrip("t"))))
        except (IndexError, ValueError):
            pass
    return sorted(pairs)


def load_energy_components(system, n, d, t):
    files = glob.glob(
        os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*_metrics.csv")
    )
    result = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "energy_j" not in df.columns or "total_iterations" not in df.columns:
            continue
        for _, row in df.iterrows():
            func  = row["function"]
            iters = row["total_iterations"]
            if iters <= 0 or not pd.notna(row["energy_j"]):
                continue
            total     = float(row["energy_j"]) / iters
            comps     = {"total": total}
            accounted = 0.0
            for name in ["cpu_energy", "memory_energy"] + ACCEL_NAMES:
                col = f"{name}_j"
                if col in df.columns and pd.notna(row[col]):
                    val = float(row[col]) / iters
                    comps[name] = val
                    accounted  += val
                else:
                    comps[name] = 0.0
            comps["others"] = max(0.0, total - accounted)
            result[func] = comps
    return result


def load_power_components(system, n, d, t):
    files = glob.glob(
        os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*_metrics.csv")
    )
    result = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "energy_j" not in df.columns or "energy_us" not in df.columns:
            continue
        for _, row in df.iterrows():
            func   = row["function"]
            dur_us = row["energy_us"]
            if not pd.notna(dur_us) or float(dur_us) <= 0:
                continue
            dur_s = float(dur_us) / 1e6
            if not pd.notna(row["energy_j"]):
                continue
            total_w   = float(row["energy_j"]) / dur_s
            comps     = {"total": total_w}
            accounted = 0.0
            for name in ["cpu_energy", "memory_energy"] + ACCEL_NAMES:
                col = f"{name}_j"
                if col in df.columns and pd.notna(row[col]):
                    val = float(row[col]) / dur_s
                    comps[name] = val
                    accounted  += val
                else:
                    comps[name] = 0.0
            comps["others"] = max(0.0, total_w - accounted)
            result[func] = comps
    return result


def make_plot(N, T, kind, all_systems, all_metric_data, d_values, metric):
    cpu_systems = sorted(
        s for s in all_systems if not _has_gpu_func(all_metric_data[s])
    )

    if kind == "cpu":
        systems    = cpu_systems
        func_order = CPU_FUNCTION_ORDER
        combo_set  = set()
        for sys_tc in systems:
            for d_data in all_metric_data[sys_tc].values():
                for func in d_data:
                    combo_set.add((func, sys_tc))

        def combo_key(c):
            func, sys_tc = c
            fi = func_order.index(func) if func in func_order else len(func_order)
            si = systems.index(sys_tc) if sys_tc in systems else len(systems)
            return (fi, si)

        ordered_combos = sorted(combo_set, key=combo_key)

    else:
        gpu_systems = sorted(
            (s for s in all_systems if _has_gpu_func(all_metric_data[s])),
            key=lambda s: GPU_GENERATION.get(s.split("/")[0], 99),
        )
        gpu_set = set()
        for sys_tc in gpu_systems:
            for d_data in all_metric_data[sys_tc].values():
                for func in d_data:
                    if "cuda" in func:
                        gpu_set.add((func, sys_tc))

        def gpu_key(c):
            func, sys_tc = c
            si = gpu_systems.index(sys_tc) if sys_tc in gpu_systems else len(gpu_systems)
            fi = GPU_FUNCTION_ORDER.index(func) if func in GPU_FUNCTION_ORDER else len(GPU_FUNCTION_ORDER)
            return (si, fi)

        ordered_combos = sorted(gpu_set, key=gpu_key)
        systems = gpu_systems

    if not ordered_combos:
        return

    scan_sys = {sys_tc for _, sys_tc in ordered_combos}
    active_components = []
    for comp in ORDERED_COMPONENTS:
        if any(
            all_metric_data[s].get(D, {}).get(func, {}).get(comp, 0.0) > 0.0
            for s in scan_sys for D in d_values
            for func, _s in ordered_combos if _s == s
        ):
            active_components.append(comp)

    if not active_components:
        return

    n_combos  = len(ordered_combos)
    bar_width = 0.8 / n_combos
    x_pos     = np.arange(len(d_values), dtype=float)

    fig_w = max(4, len(d_values) * (n_combos * bar_width + 0.6) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    for ci, (func, sys_tc) in enumerate(ordered_combos):
        offset  = (ci - (n_combos - 1) / 2) * bar_width
        hatch   = HATCHES[ci % len(HATCHES)]
        bottoms = np.zeros(len(d_values))

        for comp in active_components:
            heights = np.array([
                all_metric_data[sys_tc].get(D, {}).get(func, {}).get(comp, 0.0)
                for D in d_values
            ])
            ax.bar(
                x_pos + offset, heights, width=bar_width,
                bottom=bottoms,
                color=COMPONENT_COLORS[comp],
                hatch=hatch,
                edgecolor="white" if hatch else "#555",
                linewidth=0.4,
                zorder=2,
            )
            if np.any(heights > 0):
                any_data = True
            bottoms += heights

    if not any_data:
        plt.close(fig)
        return

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(D) for D in d_values], fontsize=9)
    ax.set_xlabel("Duration  D", fontsize=10)
    _ylabel = "Average Power  (W)" if metric == "power" else "Energy  (J per iteration)"
    ax.set_ylabel(_ylabel, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    _kind_lbl   = "GPU" if kind == "gpu" else "CPU"
    _metric_lbl = "Power" if metric == "power" else "Energy"
    ax.set_title(
        f"{_kind_lbl} {_metric_lbl} Breakdown — N={N} states, T={T:,}",
        fontsize=10,
    )

    comp_handles = [
        mpatches.Patch(facecolor=COMPONENT_COLORS[c], edgecolor="#555",
                       linewidth=0.6, label=COMPONENT_LABELS[c])
        for c in active_components
    ]
    spacer = mpatches.Patch(visible=False, label="")
    combo_handles = [
        mpatches.Patch(
            facecolor="#dddddd", edgecolor="#555",
            hatch=HATCHES[ci % len(HATCHES)], linewidth=0.6,
            label=(
                f"{FUNCTION_LABELS.get(func, func)}"
                f" — {SYSTEM_LABELS.get(sys_tc.split('/')[0], sys_tc.split('/')[0])}"
            ),
        )
        for ci, (func, sys_tc) in enumerate(ordered_combos)
    ]
    ax.legend(
        comp_handles + [spacer] + combo_handles,
        [h.get_label() for h in comp_handles + [spacer] + combo_handles],
        fontsize=7.5, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
        title="Components  /  Algorithms",
        title_fontsize=8,
    )

    out_dir  = os.path.join(OUT_ROOT, metric, kind)
    out_path = os.path.join(out_dir, f"{kind}_{N}s_{T}t_{metric}.pdf")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def _apply_accel_filter(comps, sname):
    """Return a filtered copy of comps with excluded accel channels removed from total."""
    excluded = SYSTEM_ACCEL_EXCLUDE.get(sname, set())
    if not excluded:
        return comps
    out = dict(comps)
    excl_sum = sum(comps.get(e, 0.0) for e in excluded)
    # zero out excluded channels
    for e in excluded:
        out[e] = 0.0
    # adjust total and others
    out["total"] = max(0.0, comps.get("total", 0.0) - excl_sum)
    out["others"] = max(0.0, comps.get("others", 0.0))
    return out


def make_combined_plot(N, T, all_systems, all_metric_data, d_values, metric):
    """Combined EPYC7A53 + MI250X energy/power plot in one figure."""
    focus = [s for s in all_systems if s.split("/")[0] in {"epyc-7a53", "mi250x"}]
    if not focus:
        return

    # Build ordered (func, sys_tc) combos present in data.
    # HSMMLearn_CPP is used only as the normalisation reference, not plotted.
    _SKIP_PLOT = {"HSMMLearn_CPP"}
    ordered_combos = []
    seen = set()
    for func in COMBINED_FUNC_ORDER:
        if func in _SKIP_PLOT:
            continue
        for sys_tc in focus:
            if (func, sys_tc) in seen:
                continue
            if any(func in all_metric_data[sys_tc].get(D, {}) for D in d_values):
                ordered_combos.append((func, sys_tc))
                seen.add((func, sys_tc))

    if not ordered_combos:
        return

    # Determine active components across all combos (after filtering)
    active_components = []
    for comp in ORDERED_COMPONENTS:
        if comp in {e for exc in SYSTEM_ACCEL_EXCLUDE.values() for e in exc}:
            # Check if any focus system actually keeps this component
            kept = False
            for func, sys_tc in ordered_combos:
                sname = sys_tc.split("/")[0]
                if comp not in SYSTEM_ACCEL_EXCLUDE.get(sname, set()):
                    if any(
                        all_metric_data[sys_tc].get(D, {}).get(func, {}).get(comp, 0.0) > 0.0
                        for D in d_values
                    ):
                        kept = True
                        break
            if not kept:
                continue
        else:
            if not any(
                all_metric_data[sys_tc].get(D, {}).get(func, {}).get(comp, 0.0) > 0.0
                for func, sys_tc in ordered_combos for D in d_values
            ):
                continue
        active_components.append(comp)

    if not active_components:
        return

    n_combos  = len(ordered_combos)
    bar_width = 0.55 / n_combos
    x_pos     = np.arange(len(d_values), dtype=float)

    # --- Compute per-(D) Base-1C total for normalisation -------------------
    # We use Base-1C on EPYC-7A53 (first CPU focus system that has it).
    _base_ref = {}   # D -> absolute total J (Base-1C on EPYC-7A53)
    _epyc = next((s for s in focus if s.split("/")[0] == "epyc-7a53"), None)
    for D in d_values:
        ref_comps = all_metric_data.get(_epyc, {}).get(D, {}).get("HSMMLearn_CPP", {}) if _epyc else {}
        ref_total = _apply_accel_filter(ref_comps, "epyc-7a53").get("total", 0.0) if ref_comps else 0.0
        _base_ref[D] = ref_total if ref_total > 0 else np.nan

    fig_w = max(4, len(d_values) * (n_combos * bar_width + 0.4) + 4)
    fig, ax = plt.subplots(figsize=(6, 5.5))

    bar_tops = {}   # (ci, di) -> top of bar (relative units) for annotation
    any_data = False
    for ci, (func, sys_tc) in enumerate(ordered_combos):
        sname   = sys_tc.split("/")[0]
        offset  = (ci - (n_combos - 1) / 2) * bar_width
        bottoms = np.zeros(len(d_values))

        for comp in active_components:
            abs_heights = np.array([
                _apply_accel_filter(
                    all_metric_data[sys_tc].get(D, {}).get(func, {}), sname
                ).get(comp, 0.0)
                for D in d_values
            ])
            # Normalise to Base-1C reference
            ref = np.array([_base_ref[D] for D in d_values])
            heights = np.where(ref > 0, abs_heights / ref, 0.0)
            ax.bar(
                x_pos + offset, heights, width=bar_width,
                bottom=bottoms,
                color=COMPONENT_COLORS[comp],
                edgecolor="#555",
                linewidth=0.4,
                zorder=2,
            )
            if np.any(heights > 0):
                any_data = True
            bottoms += heights

        # Store bar top for annotation
        for di in range(len(d_values)):
            bar_tops[(ci, di)] = bottoms[di]

    if not any_data:
        plt.close(fig)
        return

    # Annotate absolute values on top of each bar
    _unit = "W" if metric == "power" else "J"
    for ci, (func, sys_tc) in enumerate(ordered_combos):
        sname  = sys_tc.split("/")[0]
        offset = (ci - (n_combos - 1) / 2) * bar_width
        for di, D in enumerate(d_values):
            abs_total = _apply_accel_filter(
                all_metric_data[sys_tc].get(D, {}).get(func, {}), sname
            ).get("total", 0.0)
            if abs_total <= 0:
                continue
            top = bar_tops[(ci, di)]
            if abs_total < 1.0:
                lbl = f"{abs_total*1000:.0f}m{_unit}"
            elif abs_total < 1000:
                lbl = f"{abs_total:.1f}{_unit}"
            else:
                lbl = f"{abs_total/1000:.1f}k{_unit}"
            ax.text(
                x_pos[di] + offset, top + 0.003,
                lbl, ha="center", va="bottom",
                fontsize=15, rotation=90, color="#222",
            )

    # Per-bar sub-labels (func name) as x-ticks; D group labels between them and xlabel
    all_bar_pos    = []
    all_bar_labels = []
    for di in range(len(d_values)):
        for ci, (func, sys_tc) in enumerate(ordered_combos):
            offset = (ci - (n_combos - 1) / 2) * bar_width
            all_bar_pos.append(x_pos[di] + offset)
            all_bar_labels.append(FUNCTION_LABELS.get(func, func))
    if bar_tops:
        y_max = max(bar_tops.values())
        ax.set_ylim(0, y_max * 1.6)

    ax.set_xticks(all_bar_pos)
    ax.set_xticklabels(all_bar_labels, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis="x", length=0)

    # D group labels directly below the rotated func labels (above xlabel)
    for di, D in enumerate(d_values):
        ax.annotate(
            str(D),
            xy=(x_pos[di], 0), xycoords=("data", "axes fraction"),
            xytext=(0, -58), textcoords="offset points",
            ha="center", va="top", fontsize=14, annotation_clip=False,
        )
    ax.set_xlabel("Duration  D", fontsize=15, labelpad=14)

    _ylabel = "Relative energy  (over Base-1C)" if metric == "energy" else "Relative power  (vs Base-1C)"
    ax.set_ylabel(_ylabel, fontsize=15)
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="y", labelsize=14)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    comp_handles = [
        mpatches.Patch(facecolor=COMPONENT_COLORS[c], edgecolor="#555",
                       linewidth=0.6, label=COMPONENT_LABELS[c])
        for c in active_components
    ]
    ax.legend(
        comp_handles,
        [h.get_label() for h in comp_handles],
        fontsize=14.5, loc="upper left", ncol=2,
    )

    out_dir  = os.path.join(OUT_ROOT, metric, "combined")
    out_path = os.path.join(out_dir, f"combined_{N}s_{T}t_{metric}.pdf")
    os.makedirs(out_dir, exist_ok=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.22)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states",         "-s", type=int, default=None)
    parser.add_argument("--timesteps",      "-t", type=int, default=None)
    parser.add_argument("--metric",         choices=["energy", "power", "both"],
                        default="both",
                        help="Which metric to plot (default: both)")
    parser.add_argument("--all-toolchains", action="store_true")
    parser.add_argument("--durations", "-d", type=int, nargs="+", default=[100, 1000],
                        help="Restrict to these D values (e.g. -d 100 1000)")
    args = parser.parse_args()

    metrics = ["energy", "power"] if args.metric == "both" else [args.metric]

    all_systems = _filter_systems(discover_systems(), args.all_toolchains)
    if not all_systems:
        sys.exit("Error: no metrics results found.")

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
                os.path.join(RESULTS_ROOT, sys_tc, f"{N}s_*_{T}t_*_metrics.csv")
            ):
                parts = os.path.basename(f).split("_")
                try:
                    d_set.add(int(parts[1].rstrip("d")))
                except (IndexError, ValueError):
                    pass
        if not d_set:
            continue
        d_values = sorted(d for d in d_set if d in args.durations)
        if not d_values:
            continue

        if "energy" in metrics:
            all_energy = {
                sys_tc: {D: load_energy_components(sys_tc, N, D, T) for D in d_values}
                for sys_tc in all_systems
            }
            for kind in ("cpu", "gpu"):
                make_plot(N, T, kind, all_systems, all_energy, d_values, metric="energy")
            make_combined_plot(N, T, all_systems, all_energy, d_values, metric="energy")

        if "power" in metrics:
            all_power = {
                sys_tc: {D: load_power_components(sys_tc, N, D, T) for D in d_values}
                for sys_tc in all_systems
            }
            for kind in ("cpu", "gpu"):
                make_plot(N, T, kind, all_systems, all_power, d_values, metric="power")
            make_combined_plot(N, T, all_systems, all_power, d_values, metric="power")


if __name__ == "__main__":
    main()
