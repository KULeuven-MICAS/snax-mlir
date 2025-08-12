import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
with open("summary.txt") as f:
    lines = f.readlines()

# Parse data
records = []
for line in lines:
    parts = line.strip().split()
    if len(parts) != 2:
        continue
    filename, cycles = parts
    cycles = int(cycles)
    _match = re.match(r"matmul_m(\d+)_n(\d+)_k(\d+)_layout(\w+)_sys(\d+)_traces\.json", filename)
    if _match:
        m = int(_match.group(1))
        n = int(_match.group(2))
        k = int(_match.group(3))
        layout = _match.group(4)
        system = f"sys{_match.group(5)}"
        performance = m * n * k / cycles
        records.append(
            {
                "Layout": layout,
                "system": system,
                "m": m,
                "n": n,
                "k": k,
                "cycles": cycles,
                "performance": performance,
            }
        )

# Create DataFrame
df = pd.DataFrame(records)

# Average over number of channels (if more than one channel config is present)
avg_df = df.groupby(["Layout", "system"], as_index=False)["performance"].mean()

layout_mapping = {
    "banked": "Banked",
    "cyclic": "Tiled",
    "default": "Row-Major",
}

avg_df["Layout"] = avg_df["Layout"].map(layout_mapping)

# Plot
plt.figure(figsize=(5, 3))
sns.lineplot(data=avg_df, x="system", y="performance", style="Layout", markers=True, dashes=True, palette=["#0b5394ff"])
plt.ylabel("Effective OPS/cycle")
plt.xlabel("FIFO Depth")
xticks = sorted(avg_df["system"].unique())
plt.xticks(ticks=xticks, labels=[str(x)[-1] for x in xticks])
plt.tight_layout()
plt.savefig("plot.png", dpi=600)
