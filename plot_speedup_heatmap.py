import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

cpp = agg[agg["function"] == "HSMMLearn_CPP"].rename(columns={"elapsed_s": "cpp_s"})
omp = agg[agg["function"] == "HSMMLearn_OMP"].rename(columns={"elapsed_s": "omp_s"})

merged = pd.merge(
    cpp[["n_states", "timesteps", "max_duration", "cpp_s"]],
    omp[["n_states", "timesteps", "max_duration", "omp_s"]],
    on=["n_states", "timesteps", "max_duration"],
)
merged["speedup"] = merged["cpp_s"] / merged["omp_s"]

timestep_values = sorted(merged["timesteps"].unique())
states_vals     = sorted(merged["n_states"].unique())
dur_vals        = sorted(merged["max_duration"].unique(), reverse=True)  # y: high→low

fig, axes = plt.subplots(
    1, len(timestep_values),
    figsize=(4.5 * len(timestep_values), 4),
)
if len(timestep_values) == 1:
    axes = [axes]

vmin = merged["speedup"].min()
vmax = merged["speedup"].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "YlOrRd"

for ax, T in zip(axes, timestep_values):
    sub = merged[merged["timesteps"] == T]

    # Build matrix: rows = durations (desc), cols = states (asc)
    matrix = np.full((len(dur_vals), len(states_vals)), np.nan)
    for _, row in sub.iterrows():
        ri = dur_vals.index(row["max_duration"])
        ci = states_vals.index(row["n_states"])
        matrix[ri, ci] = row["speedup"]

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    # Annotate cells
    for ri in range(len(dur_vals)):
        for ci in range(len(states_vals)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                color = "black" if val < (vmin + (vmax - vmin) * 0.65) else "white"
                ax.text(ci, ri, f"{val:.1f}x", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(states_vals)))
    ax.set_xticklabels(states_vals, fontsize=9)
    ax.set_yticks(range(len(dur_vals)))
    ax.set_yticklabels(dur_vals, fontsize=9)
    ax.set_xlabel("States (N)", fontsize=10)
    ax.set_ylabel("Max Duration (D)", fontsize=10)
    ax.set_title(f"T = {T:,} timesteps", fontsize=11, fontweight="bold")

# Shared colorbar
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes, shrink=0.8, pad=0.02,
)
cbar.set_label("Speedup OMP / CPP  (×)", fontsize=10)

fig.suptitle("OMP Speedup over CPP Baseline — Leonardo", fontsize=13, fontweight="bold")

out_path = "logs/leonardo/speedup_heatmap.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
