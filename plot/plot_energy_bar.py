#!/usr/bin/env python3
"""
plot_energy_bar.py — energy breakdown stacked bar charts.

Each bar represents one (function, system/toolchain) combo.
Bars are stacked by energy component:
  - cpu_energy
  - memory_energy
  - accel0_energy … accel3_energy  (GPU nodes, when data is present)
  - others  =  total − (cpu + mem + accels)
Total bar height = total energy (J per iteration).

Generates two plots per (N, T) pair:
  bars/cpu/cpu_{N}s_{T}t_energy.png    — CPU systems
  bars/gpu/gpu_{N}s_{T}t_energy.png    — GPU systems

Usage:
  python plot/plot_energy_bar.py              # all (N, T) pairs
  python plot/plot_energy_bar.py --states 75
  python plot/plot_energy_bar.py --states 75 --timesteps 100000
  python plot/plot_energy_bar.py --all-toolchains
"""

import argparse
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

DEFAULT_TOOLCHAINS = {
    "a100":      "cuda",
    "b200":      "cuda",
    "h100":      "cuda",
    "mi250x":    "cray",
    "epyc-7763": "cray",
    "xeon8480":  "gnu",
}

GPU_GENERATION = {
    "a100":         0,
    "mi250x":       1,
    "h100":         2,
    "gh200-hopper": 3,
    "mi300x":       4,
    "b200":         5,
}

CPU_FUNCTION_ORDER = [
    "HSMMLearn_CPP",
    "HSMMLearn_OMP",
    "decode_tensor_viterbi_cpp",
    "decode_tensor_viterbi_omp",
]
GPU_FUNCTION_ORDER = [
    "decode_tensor_viterbi_cuda",
]
# CPU functions shown as reference bars inside GPU energy plots
GPU_REF_FUNCTIONS = ["HSMMLearn_OMP", "decode_tensor_viterbi_omp"]
FUNCTION_LABELS = {
    "HSMMLearn_CPP":              "HSMMLearn C++",
    "HSMMLearn_OMP":              "HSMMLearn OMP",
    "decode_tensor_viterbi_cpp":  "Tensor C++",
    "decode_tensor_viterbi_omp":  "Tensor OMP",
    "decode_tensor_viterbi_cuda": "Tensor GPU",
}

# ── Energy component config ───────────────────────────────────────────────────

ACCEL_NAMES = [f"accel{i}_energy" for i in range(4)]
ORDERED_COMPONENTS = ["cpu_energy", "memory_energy"] + ACCEL_NAMES + ["others"]

COMPONENT_COLORS = {
    "cpu_energy":    "#4e79a7",   # blue
    "memory_energy": "#59a14f",   # green
    "accel0_energy": "#f28e2b",   # orange
    "accel1_energy": "#e15759",   # red
    "accel2_energy": "#b07aa1",   # purple
    "accel3_energy": "#ff9da7",   # pink
    "others":        "#bab0ac",   # grey
}
COMPONENT_LABELS = {
    "cpu_energy":    "CPU",
    "memory_energy": "Memory",
    "accel0_energy": "Accel 0",
    "accel1_energy": "Accel 1",
    "accel2_energy": "Accel 2",
    "accel3_energy": "Accel 3",
    "others":        "Others",
}

HATCHES = ["", "///", "\\\\", "xxx", "...", "|||", "---", "+++"]


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


def _has_gpu_func(sys_tc_energy):
    """True if any D-value entry for this system has a cuda function."""
    for d_data in sys_tc_energy.values():        # d_data = {func: {comp: value}}
        if any("cuda" in func for func in d_data):
            return True
    return False


def discover_systems():
    """Systems that have at least one _metrics.csv file."""
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
        fname = os.path.basename(p)
        parts = fname.split("_")
        try:
            n = int(parts[0].rstrip("s"))
            t = int(parts[2].rstrip("t"))
            pairs.add((n, t))
        except (IndexError, ValueError):
            pass
    return sorted(pairs)


def load_energy_components(system, n, d, t):
    """Return {func: {component: j_per_iter}} including 'total' and 'others'."""
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*_metrics.csv"))
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
            if iters <= 0:
                continue
            raw_total = row["energy_j"]
            if not pd.notna(raw_total):
                continue
            total     = float(raw_total) / iters
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
    """Return {func: {component: watts}} where watts = energy_j / (energy_us / 1e6)."""
    files = glob.glob(os.path.join(RESULTS_ROOT, system, f"{n}s_{d}d_{t}t_*_metrics.csv"))
    result = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "energy_j" not in df.columns or "energy_us" not in df.columns:
            continue
        for _, row in df.iterrows():
            func    = row["function"]
            dur_us  = row["energy_us"]
            if not pd.notna(dur_us) or float(dur_us) <= 0:
                continue
            dur_s   = float(dur_us) / 1e6
            raw_j   = row["energy_j"]
            if not pd.notna(raw_j):
                continue
            total_w = float(raw_j) / dur_s
            comps   = {"total": total_w}
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


# ── Per-kind plot ─────────────────────────────────────────────────────────────

def make_plot(N, T, kind, all_systems, all_energy, d_values, metric="energy"):
    """
    kind   : 'cpu' or 'gpu'
    metric : 'energy' (J/iter) or 'power' (W average)
    """
    cpu_systems = sorted(s for s in all_systems if not _has_gpu_func(all_energy[s]))

    if kind == "cpu":
        systems = cpu_systems

        combo_set = set()
        for sys_tc in systems:
            for d_data in all_energy[sys_tc].values():
                for func in d_data:
                    combo_set.add((func, sys_tc))

        def combo_key(c):
            func, sys_tc = c
            si = systems.index(sys_tc) if sys_tc in systems else len(systems)
            fi = CPU_FUNCTION_ORDER.index(func) if func in CPU_FUNCTION_ORDER else len(CPU_FUNCTION_ORDER)
            return (si, fi)

        ordered_combos = sorted(combo_set, key=combo_key)
    else:
        gpu_systems = sorted(
            (s for s in all_systems if _has_gpu_func(all_energy[s])),
            key=lambda s: GPU_GENERATION.get(s.split("/")[0], 99),
        )

        # GPU function combos (cuda only)
        gpu_combo_set = set()
        for sys_tc in gpu_systems:
            for d_data in all_energy[sys_tc].values():
                for func in d_data:
                    if "cuda" in func:
                        gpu_combo_set.add((func, sys_tc))

        def gpu_combo_key(c):
            func, sys_tc = c
            si = gpu_systems.index(sys_tc) if sys_tc in gpu_systems else len(gpu_systems)
            fi = GPU_FUNCTION_ORDER.index(func) if func in GPU_FUNCTION_ORDER else len(GPU_FUNCTION_ORDER)
            return (si, fi)

        # CPU reference combos (HSMMLearn_OMP, decode_tensor_viterbi_omp)
        cpu_ref_set = set()
        for sys_tc in cpu_systems:
            for d_data in all_energy[sys_tc].values():
                for func in d_data:
                    if func in GPU_REF_FUNCTIONS:
                        cpu_ref_set.add((func, sys_tc))

        def cpu_ref_key(c):
            func, sys_tc = c
            si = cpu_systems.index(sys_tc) if sys_tc in cpu_systems else len(cpu_systems)
            fi = GPU_REF_FUNCTIONS.index(func) if func in GPU_REF_FUNCTIONS else len(GPU_REF_FUNCTIONS)
            return (si, fi)

        ordered_combos = (
            sorted(gpu_combo_set, key=gpu_combo_key)
            + sorted(cpu_ref_set, key=cpu_ref_key)
        )

    if not ordered_combos:
        return

    # Which components are actually non-zero across all combos in this plot?
    scan_systems = {sys_tc for _, sys_tc in ordered_combos}
    active_components = []
    for comp in ORDERED_COMPONENTS:
        found = False
        for sys_tc in scan_systems:
            for D_val in d_values:
                for func_data in all_energy[sys_tc].get(D_val, {}).values():
                    if func_data.get(comp, 0.0) > 0.0:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            active_components.append(comp)

    if not active_components:
        return

    n_combos  = len(ordered_combos)
    bar_width = 0.8 / n_combos
    x_pos     = np.arange(len(d_values), dtype=float)

    fig_w = max(8, len(d_values) * (n_combos * bar_width + 0.6) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    any_data = False
    for ci, (func, sys_tc) in enumerate(ordered_combos):
        offset  = (ci - (n_combos - 1) / 2) * bar_width
        hatch   = HATCHES[ci % len(HATCHES)]
        bottoms = np.zeros(len(d_values))

        for comp in active_components:
            heights = np.array([
                all_energy[sys_tc].get(D_val, {}).get(func, {}).get(comp, 0.0)
                for D_val in d_values
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
    ax.set_xticklabels([f"D = {D}" for D in d_values], fontsize=9)
    ax.set_xlabel("Duration  D", fontsize=9)
    _ylabel = "Average Power  (W)" if metric == "power" else "Energy  (J per iteration)"
    ax.set_ylabel(_ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    _metric_label = "Power" if metric == "power" else "Energy"
    if kind == "gpu":
        ax.set_title(
            f"GPU {_metric_label} Breakdown  (+ CPU OMP reference)\nN = {N} states,  T = {T:,}",
            fontsize=10,
        )
    else:
        ax.set_title(
            f"CPU {_metric_label} Breakdown\nN = {N} states,  T = {T:,}",
            fontsize=10,
        )

    # Two-level legend: component colours + function/system hatching
    comp_handles = [
        mpatches.Patch(facecolor=COMPONENT_COLORS[c], edgecolor="#555",
                       linewidth=0.6, label=COMPONENT_LABELS[c])
        for c in active_components
    ]
    spacer = mpatches.Patch(visible=False, label="")
    combo_handles = [
        mpatches.Patch(
            facecolor="#dddddd", edgecolor="#555",
            hatch=HATCHES[ci % len(HATCHES)],
            linewidth=0.6,
            label=f"{FUNCTION_LABELS.get(func, func)} — {sys_tc}",
        )
        for ci, (func, sys_tc) in enumerate(ordered_combos)
    ]
    all_handles = comp_handles + [spacer] + combo_handles
    ax.legend(
        all_handles, [h.get_label() for h in all_handles],
        fontsize=7.5, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
        title="Components  /  Systems",
        title_fontsize=8,
    )

    _msuffix = "power" if metric == "power" else "energy"
    if kind == "gpu":
        out_dir  = os.path.join(OUT_ROOT, "gpu")
        out_path = os.path.join(out_dir, f"gpu_{N}s_{T}t_{_msuffix}.png")
    else:
        out_dir  = os.path.join(OUT_ROOT, "cpu")
        out_path = os.path.join(out_dir, f"cpu_{N}s_{T}t_{_msuffix}.png")
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
    parser.add_argument("--all-toolchains", action="store_true",
                        help="Show all toolchains; by default only the "
                             "default toolchain per system is shown.")
    args = parser.parse_args()

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
        d_values = sorted(d_set)

        all_energy = {}
        for sys_tc in all_systems:
            all_energy[sys_tc] = {
                D_val: load_energy_components(sys_tc, N, D_val, T)
                for D_val in d_values
            }

        make_plot(N, T, "cpu", all_systems, all_energy, d_values, metric="energy")
        make_plot(N, T, "gpu", all_systems, all_energy, d_values, metric="energy")

        all_power = {}
        for sys_tc in all_systems:
            all_power[sys_tc] = {
                D_val: load_power_components(sys_tc, N, D_val, T)
                for D_val in d_values
            }
        make_plot(N, T, "cpu", all_systems, all_power, d_values, metric="power")
        make_plot(N, T, "gpu", all_systems, all_power, d_values, metric="power")


if __name__ == "__main__":
    main()
