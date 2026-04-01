import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load all CSVs ─────────────────────────────────────────────────────────────
csv_files = glob.glob("logs/leonardo/**/*.csv", recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV files found under logs/leonardo/")

df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Mean elapsed per (function, n_states, timesteps, max_duration)
agg = (
    df.groupby(["function", "n_states", "timesteps", "max_duration"], as_index=False)
    ["elapsed_s"].mean()
)

timestep_values = sorted(agg["timesteps"].unique())
functions       = sorted(agg["function"].unique())

COLORS = {
    "HSMMLearn_CPP": "#4C72B0",
    "HSMMLearn_OMP": "#DD8452",
}
FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(
    1, len(timestep_values),
    figsize=(6 * len(timestep_values), 6),
    sharey=False,
)
if len(timestep_values) == 1:
    axes = [axes]

for ax, T in zip(axes, timestep_values):
    sub = agg[agg["timesteps"] == T].copy()

    # x-axis: each unique (n_states, max_duration) combo, sorted
    combos = (
        sub[["n_states", "max_duration"]]
        .drop_duplicates()
        .sort_values(["n_states", "max_duration"])
    )
    x_labels = [f"{int(r.n_states)}s\n{int(r.max_duration)}d" for _, r in combos.iterrows()]
    n_combos  = len(x_labels)
    n_funcs   = len(functions)
    width     = 0.8 / n_funcs
    x         = np.arange(n_combos)

    for fi, func in enumerate(functions):
        fsub   = sub[sub["function"] == func]
        heights = []
        for _, row in combos.iterrows():
            match = fsub[
                (fsub["n_states"] == row.n_states) &
                (fsub["max_duration"] == row.max_duration)
            ]
            heights.append(match["elapsed_s"].values[0] if len(match) else 0.0)

        color  = COLORS.get(func, FALLBACK_COLORS[fi % len(FALLBACK_COLORS)])
        offset = (fi - (n_funcs - 1) / 2) * width
        bars   = ax.bar(x + offset, heights, width=width * 0.95,
                        label=func, color=color, edgecolor="white", linewidth=0.5)

        # value labels on bars
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 1.02,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("States × Max Duration", fontsize=10)
    ax.set_ylabel("Mean elapsed (s)", fontsize=10)
    ax.set_title(f"T = {T:,} timesteps", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

fig.suptitle("Viterbi Baseline Benchmark — Leonardo", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()

out_path = "logs/leonardo/benchmark_plot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
