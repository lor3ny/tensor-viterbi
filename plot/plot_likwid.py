#!/usr/bin/env python3
"""
plot_likwid.py — Parsa e visualizza metriche raw LIKWID (tutti i group).

Utilizzo:
    python plot_likwid.py --system leonardo --toolchain xeon4840/gnu
    python plot_likwid.py --system leonardo --toolchain xeon4840/gnu --stem report
"""

import re
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PERF_GROUPS = ["FLOPS_DP", "MEM", "L3", "L2", "TMA", "BRANCH"]

# All available metrics and their display config
METRIC_CONFIG = {
    "DP_MFLOPS":        ("DP FLOPS\n[MFLOP/s]",       1.0),
    "AVX_DP_MFLOPS":    ("AVX DP FLOPS\n[MFLOP/s]",   1.0),
    "AVX512_DP_MFLOPS": ("AVX512 DP FLOPS\n[MFLOP/s]",1.0),
    "Vec_ratio_pct":    ("Vectorization\n[%]",         1.0),
    "CPI":              ("CPI",                        1.0),
    "IPC":              ("IPC",                        1.0),
    "Runtime_s":        ("Runtime\n[s]",               1.0),
    "Bad_Spec_pct":     ("Bad Speculation\n[%]",       1.0),
    "MEM_BW_total":     ("DRAM BW\n[MBytes/s]",        1.0),
    "MEM_BW_read":      ("DRAM Read BW\n[MBytes/s]",   1.0),
    "MEM_BW_write":     ("DRAM Write BW\n[MBytes/s]",  1.0),
    "MEM_vol_GB":       ("DRAM Volume\n[GB]",          1.0),
    "L3_BW_total":      ("L3 BW\n[MBytes/s]",          1.0),
    "L3_BW_load":       ("L3 Load BW\n[MBytes/s]",     1.0),
    "L3_BW_evict":      ("L3 Evict BW\n[MBytes/s]",    1.0),
    "L3_vol_GB":        ("L3 Volume\n[GB]",            1.0),
    "L2_BW_total":      ("L2 BW\n[MBytes/s]",          1.0),
    "L2_BW_load":       ("L2 Load BW\n[MBytes/s]",     1.0),
    "L2_BW_evict":      ("L2 Evict BW\n[MBytes/s]",    1.0),
    "L2_vol_GB":        ("L2 Volume\n[GB]",            1.0),
    "Branch_rate":      ("Branch Rate",                1.0),
    "Branch_mispredict_rate":   ("Branch Mispredict\nRate",  1.0),
    "Branch_mispredict_ratio":  ("Branch Mispredict\nRatio", 1.0),
    "Instr_per_branch":         ("Instr per\nBranch",        1.0),
}

DEFAULT_METRICS = ["Vec_ratio_pct", "MEM_vol_GB", "Branch_mispredict_ratio"]

HATCHES = ["", "//", "xx", "\\\\"]
matplotlib.rcParams["hatch.linewidth"] = 0.4

# ─── Parser ──────────────────────────────────────────────────────────────────

def _parse_likwid_csv(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """
    Parse a single LIKWID CSV file.
    Returns (raw_events, derived_metrics) as {name: float} dicts.
    """
    raw: dict[str, float] = {}
    metrics: dict[str, float] = {}
    mode = None  # "raw" | "metric" | None

    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue

        tag = parts[0]

        # Section header
        if tag == "TABLE":
            mode = "raw" if any("Raw" in p for p in parts) else \
                "metric" if any("Metric" in p for p in parts) else None
            continue

        # Column headers / non-data rows
        if tag in ("STRUCT", "Region Info", "Event", "Metric"):
            continue

        if mode == "raw":
            # rows: EVENT_NAME, COUNTER, VALUE
            if len(parts) >= 3:
                try:
                    raw[tag] = float(parts[2])
                except ValueError:
                    pass

        elif mode == "metric":
            # rows: Metric Name, VALUE
            if len(parts) >= 2:
                try:
                    metrics[tag] = float(parts[1])
                except ValueError:
                    pass

    #print(f"  [debug] {path.name}: raw={list(raw)}, metrics={list(metrics)}")
    return raw, metrics


def parse_files(outdir: Path, version: str) -> dict:
    """Parse all per-group LIKWID CSV files and return a combined metrics dict."""

    def read(group: str) -> tuple[dict, dict]:
        path = outdir / f"likwid_{version}_{group}.csv"
        #print(f"  [debug] looking for: {path} — exists: {path.exists()}")
        return _parse_likwid_csv(path) if path.exists() else ({}, {})

    result: dict[str, float | None] = {}

    # ── FLOPS_DP ──────────────────────────────────────────────────────────────
    raw, met = read("FLOPS_DP")
    result["DP_MFLOPS"]        = met.get("DP [MFLOP/s]")
    result["AVX_DP_MFLOPS"]    = met.get("AVX DP [MFLOP/s]")
    result["AVX512_DP_MFLOPS"] = met.get("AVX512 DP [MFLOP/s]")
    result["Scalar_MUOPS"]     = met.get("Scalar [MUOPS/s]")
    result["Packed_MUOPS"]     = met.get("Packed [MUOPS/s]")
    result["Vec_ratio_pct"]    = met.get("Vectorization ratio")
    result["FP_SCALAR"]        = raw.get("FP_ARITH_INST_RETIRED_SCALAR_DOUBLE")
    result["FP_128B"]          = raw.get("FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE")
    result["FP_256B"]          = raw.get("FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE")
    result["FP_512B"]          = raw.get("FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE")
    result["CPI"]              = met.get("CPI")
    result["IPC"]              = met.get("IPC")
    result["Runtime_s"]        = met.get("Runtime (RDTSC) [s]")

    # ── TMA ───────────────────────────────────────────────────────────────────
    _, met = read("TMA")
    result["Bad_Spec_pct"] = met.get("Bad Speculation (PMC) [%]")
    # TMA also carries CPI/IPC — use as fallback if FLOPS_DP didn't have them
    result["CPI"] = result["CPI"] or met.get("CPI")
    result["IPC"] = result["IPC"] or met.get("IPC")

    # ── MEM ───────────────────────────────────────────────────────────────────
    _, met = read("MEM")
    result["MEM_BW_total"]  = met.get("Memory bandwidth [MBytes/s]")
    result["MEM_BW_read"]   = met.get("Memory read bandwidth [MBytes/s]")
    result["MEM_BW_write"]  = met.get("Memory write bandwidth [MBytes/s]")
    result["MEM_vol_GB"]    = met.get("Memory data volume [GBytes]")

    # ── L3 ────────────────────────────────────────────────────────────────────
    _, met = read("L3")
    result["L3_BW_total"]  = met.get("L3 bandwidth [MBytes/s]") or met.get("L3 access bandwidth [MBytes/s]")
    result["L3_BW_load"]   = met.get("L3 load bandwidth [MBytes/s]")
    result["L3_BW_evict"]  = met.get("L3 evict bandwidth [MBytes/s]")
    result["L3_vol_GB"]    = met.get("L3 data volume [GBytes]") or met.get("L3 access data volume [GBytes]")

    # ── L2 ────────────────────────────────────────────────────────────────────
    _, met = read("L2")
    result["L2_BW_total"]  = met.get("L2 bandwidth [MBytes/s]")
    result["L2_BW_load"]   = met.get("L2D load bandwidth [MBytes/s]")
    result["L2_BW_evict"]  = met.get("L2D evict bandwidth [MBytes/s]")
    result["L2_vol_GB"]    = met.get("L2 data volume [GBytes]")

    # ── BRANCH ───────────────────────────────────────────────────────────────
    _, met = read("BRANCH")
    result["Branch_rate"]             = met.get("Branch rate")
    result["Branch_mispredict_rate"]  = met.get("Branch misprediction rate")
    result["Branch_mispredict_ratio"] = met.get("Branch misprediction ratio")
    result["Instr_per_branch"]        = met.get("Instructions per branch")

    return result


# ─── Plotting ────────────────────────────────────────────────────────────────

def _make_colors(n):
    cmap  = plt.get_cmap("tab10" if n <= 10 else "tab20")
    n_map = 10 if n <= 10 else 20
    return [cmap(i / n_map) for i in range(n)]


def plot_metric_subplot(ax, data_list, metric_key, xlabel, scale, colors):
    """Un subplot per una singola metrica; le versioni sono le barre sull'asse x."""
    vals  = [d.get(metric_key) or 0.0 for d in data_list]
    width = 0.6
    vmax  = max((v * scale for v in vals if v > 0), default=1)

    for i, (val, color, hatch) in enumerate(zip(vals, colors, HATCHES)):
        scaled = val * scale
        ax.bar(i, scaled, width=width, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6, zorder=2)
        if scaled > 0:
            fmt = f"{scaled:.3f}" if scaled < 10 else f"{scaled:.1f}"
            if fmt == "0.000":
                # valore troppo piccolo per essere leggibile: annotazione verticale
                ax.text(i, vmax * 0.025, f"   <{scaled + 5e-4:.3f}",
                        ha="center", va="bottom", fontsize=10, color="black",
                        rotation=90)
            else:
                ax.text(i, scaled + vmax * 0.025, fmt,
                        ha="center", va="bottom", fontsize=10, color="black", rotation=90)

    n = len(vals)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, vmax * 1.5)      
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_all(data_list, labels, metrics, output_dir: Path, output_stem: str):
    from matplotlib.patches import Patch
    n_metrics  = len(metrics)
    n_versions = len(data_list)
    colors     = _make_colors(n_versions)

    fig = plt.figure(figsize=(2.5 * n_metrics, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, n_metrics, height_ratios=[0.05, 1])
    
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    axes = [fig.add_subplot(gs[1, i]) for i in range(n_metrics)]

    fig.suptitle("Profiling Details", fontsize=13)

    for ax, (key, xlabel, scale) in zip(axes, metrics):
        plot_metric_subplot(ax, data_list, key, xlabel, scale, colors)

    fig.canvas.draw()
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)

    handles = [
        Patch(facecolor=colors[i], hatch=HATCHES[i],
              edgecolor="black", linewidth=0.6, label=lbl)
        for i, lbl in enumerate(labels)
    ]
    ax_leg.legend(
        handles=handles,
        ncol=n_versions,
        fontsize=10.5,
        framealpha=0.85,
        loc="center",
        bbox_to_anchor=(x0, 0.9, x1 - x0, 0.0),
        bbox_transform=fig.transFigure,
        mode="expand",
        borderaxespad=0,
        handletextpad=0.2,
    )

    out = output_dir / f"{output_stem}_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[✓] Salvato: {out}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Confronta metriche LIKWID raw da tutti i .txt in una cartella."
    )
    parser.add_argument("--system",    required=True,
        help="Nome del sistema (es. leonardo)")
    parser.add_argument("--toolchain", required=True,
        help="Toolchain nel formato cpu/compiler (es. xeon4840/gnu)")
    parser.add_argument("--stem",      default=None,
        help="Prefisso custom per i file di output")
    parser.add_argument("--metrics", nargs="+",
        default=DEFAULT_METRICS,
        choices=list(METRIC_CONFIG),
        metavar="METRIC",
        help=f"Metrics to plot. Available: {', '.join(METRIC_CONFIG)}. "
            f"Default: {' '.join(DEFAULT_METRICS)}"
    )
    args = parser.parse_args()


    base_path = Path("results") / args.system / args.toolchain    
    if not base_path.is_dir():
        print(f"[✗] Cartella non trovata: {base_path}", file=sys.stderr)
        sys.exit(1)

    versions = set()
    for f in base_path.glob("likwid_*.csv"):
        name = f.stem.removeprefix("likwid_")
        for g in PERF_GROUPS:
            if name.endswith(f"_{g}"):
                versions.add(name[: -(len(g) + 1)])
                break
    versions = sorted(versions)
    if not versions:
        print(f"[✗] Nessun file CSV trovato in: {base_path}", file=sys.stderr)
        sys.exit(1)

    LABEL_MAP = {
        "baseline":     "Base-1C",
        "baseline-omp": "Base-MC",
        "cpp":          "Tens-1C",
        "omp":          "Tens-MC",
    }
    labels    = [LABEL_MAP.get(v, v) for v in versions]
    data_list = []
    for v in versions:
        d = parse_files(base_path, v)
        data_list.append(d)
        print(f"[✓] Parsed: {v}")

    toolchain_flat = args.toolchain.replace("/", "_")
    output_stem    = args.stem or f"{args.system}_{toolchain_flat}"
    metrics = [(k, *METRIC_CONFIG[k]) for k in args.metrics]
    plot_all(data_list, labels, metrics=metrics, output_dir=base_path, output_stem=output_stem)
    
if __name__ == "__main__":
    main()