import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

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
                "layout": layout,
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
avg_df = df.groupby(["layout", "system"], as_index=False)["performance"].mean()

# Plot
plt.figure(figsize=(5, 3))
sns.lineplot(data=avg_df, x="system", y="performance", hue="layout", marker="o")
plt.title("Average Performance by Layout and System")
plt.ylabel("Performance")
plt.xlabel("System")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot.svg")
