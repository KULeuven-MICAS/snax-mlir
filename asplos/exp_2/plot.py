import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open("summary_base.txt") as f:
    data = f.read()

# Parse data into DataFrame
rows = [line.split() for line in data.strip().split("\n")]
df = pd.DataFrame(rows, columns=["filename", "cycles", "status"])
df["cycles"] = df["cycles"].astype(int)

# Compute effective ops per cycle
num_ops = 10616832
df["ops_per_cycle"] = num_ops / df["cycles"]

# Set up seaborn plot
sns.set(style="whitegrid")
# plt.figure(figsize=(8, 5))
#
# # Lineplot with only markers (no connecting line)
# sns.scatterplot(
#     x=range(len(df)),
#     y="ops_per_cycle",
#     data=df,
#     s=20,  # marker size
#     marker="o",
# )
#
# plt.xlabel("Schedule Index")
# plt.ylabel("Effective OPS/cycle")
# plt.tight_layout()
# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
# plt.savefig("plot.png", dpi=600)
#
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)  # or omit and use subplots_adjust below

sns.scatterplot(
    x=range(len(df)),
    y="ops_per_cycle",
    data=df,
    s=18,
    marker="o",
    ax=ax,
)

ax.set_xlabel("Schedule Index")
ax.set_ylabel("Effective OPS/cycle")

# Remove the plot frame (bounding box)
for side in ["top", "left", "bottom", "right"]:
    ax.spines[side].set_visible(False)


# Put y-axis on the right
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.labelpad = 8  # small gap between ticks and label
ax.grid(axis="x")

# If you didn’t use constrained_layout above, do this:
# fig.tight_layout(pad=0.5)
# fig.subplots_adjust(right=0.88)  # add space on the right

plt.savefig("plot.png", dpi=600)
