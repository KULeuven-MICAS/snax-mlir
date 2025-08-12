import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from dacite import from_dict

from asplos.util.convspecs import TiledConfig

with open("tiled_resnet_layers.yaml") as f:
    yaml_data = yaml.safe_load(f)

tiled_config = from_dict(data_class=TiledConfig, data=yaml_data)

# size * 2 because of double buffering
data = {
    "Layer": [layer.layer.name for layer in tiled_config.layers],
    "I": [layer.input_tile_size() * 2 / 1024 for layer in tiled_config.layers],
    "O": [layer.output_tile_size() * 2 / 1024 for layer in tiled_config.layers],
    "W": [layer.weight_tile_size() * 2 / 1024 for layer in tiled_config.layers],
}

df = pd.DataFrame(data)

# Max values
max_i = df["I"].max()
max_o = df["O"].max()
max_w = df["W"].max()
max_total_dynamic = (df["I"] + df["O"] + df["W"]).max()

# Summary rows
dynamic_row = pd.DataFrame([{"Layer": "Shared\nMemory", "I": 0, "O": 0, "W": 0, "Total": max_total_dynamic}])
static_row = pd.DataFrame([{"Layer": "Per-Operand\nMemory", "I": max_i, "O": max_o, "W": max_w}])
df["Total"] = df["I"] + df["O"] + df["W"]

# Build final DataFrame
df_final = pd.concat([df, static_row, dynamic_row], ignore_index=True).fillna(0)

# Define custom x positions:
spacing = 0.6  # tighter spacing
gap_after_layers = 2.0  # space between last per-layer and first summary bar
x_layers = np.arange(len(df)) * spacing
x_gap = x_layers[-1] + gap_after_layers
x_summary = [x_gap, x_gap + 1.0]
x_all = np.concatenate([x_layers, x_summary])

# Plot setup
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(18 / 2, 7 / 2))

layer_width = 0.4
summary_width = 0.8

# Background shading for summary region
ax.axvspan(x_gap - summary_width, x_all[-1] + summary_width, color="lightgray", alpha=0.5)

# Plot per-layer bars
for i in range(len(df)):
    xi = x_all[i]
    ax.bar(xi, df_final.loc[i, "I"], layer_width, color="cornflowerblue")
    ax.bar(xi, df_final.loc[i, "O"], layer_width, bottom=df_final.loc[i, "I"], color="orange")
    ax.bar(xi, df_final.loc[i, "W"], layer_width, bottom=df_final.loc[i, "I"] + df_final.loc[i, "O"], color="indianred")

# Plot summary bars
idx_static = len(df)
idx_dynamic = len(df) + 1

# max(dynamic)
ax.bar(
    x_all[idx_dynamic],
    df_final.loc[idx_dynamic, "Total"],
    summary_width,
    color="goldenrod",
    hatch="//",
    label="Shared\nMemory",
)

# static
ax.bar(x_all[idx_static], df_final.loc[idx_static, "I"], summary_width, color="cornflowerblue", hatch="//")
ax.bar(
    x_all[idx_static],
    df_final.loc[idx_static, "O"],
    summary_width,
    bottom=df_final.loc[idx_static, "I"],
    color="orange",
    hatch="//",
)
ax.bar(
    x_all[idx_static],
    df_final.loc[idx_static, "W"],
    summary_width,
    bottom=df_final.loc[idx_static, "I"] + df_final.loc[idx_static, "O"],
    color="indianred",
    hatch="//",
)

# max(dynamic)
x_dyn = x_all[idx_dynamic]
y_dyn = df_final.loc[idx_dynamic, "Total"]
ax.text(x_dyn, y_dyn + 3, f"{int(y_dyn)} kB", ha="center", va="bottom", fontsize=10)

# static
x_stat = x_all[idx_static]
y_stat = df_final.loc[idx_static, "I"] + df_final.loc[idx_static, "O"] + df_final.loc[idx_static, "W"]
ax.text(x_stat, y_stat + 3, f"{int(y_stat)} kB", ha="center", va="bottom", fontsize=10)

# Compute difference
diff_kb = y_stat - y_dyn
diff_pct = diff_kb / y_stat * 100

# Threshold line
# ax.axhline(128, color="red", linestyle="--", linewidth=1.2)
ax.axvline(13, color="black", linestyle="-", linewidth=2, alpha=0.6)
# ax.text(x_all[-1] - 0.5, 130, "128 kB", color="red", fontsize=10)

# Axes and labels
ax.set_ylabel("Memory Usage (kB)")
ax.set_xticks(x_all)
ax.set_xticklabels(df_final["Layer"], rotation=60, ha="right")
ax.set_ylim(0, max(df_final["Total"].max(), max_i + max_o + max_w) + 100)

# Region annotations
# ax.text(x_layers.mean(), -102, "Per-layer Tile Sizes", ha="center", va="top", fontsize=11)
# ax.text(np.mean(x_summary), -102, "Overall Memory Requirements", ha="center", va="top", fontsize=11)

# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color="cornflowerblue", label="Inputs"),
    plt.Rectangle((0, 0), 1, 1, color="orange", label="Outputs"),
    plt.Rectangle((0, 0), 1, 1, color="indianred", label="Weights"),
    # plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.6, hatch="//", label="max(dynamic)"),
]
ax.legend(handles=handles, loc="upper left")

# plt.title("Shared Memory Allocation per Layer with Dynamic and Static Peak Comparison")
plt.tight_layout()
plt.savefig("plot.png", dpi=600)
