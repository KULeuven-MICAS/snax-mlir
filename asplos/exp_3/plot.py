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
    m = re.match(r"conv_channel(\d+)_layout(\w+)_sys(\d+)_traces\.json", filename)
    if m:
        nb_channels = int(m.group(1))
        layout = m.group(2)
        system = f"sys{m.group(3)}"
        performance = 36864 * nb_channels / cycles
        records.append(
            {
                "layout": layout,
                "system": system,
                "nb_channels": nb_channels,
                "cycles": cycles,
                "performance": performance,
            }
        )

# Create DataFrame
df = pd.DataFrame(records)

# Average over number of channels (if more than one channel config is present)
avg_df = df.groupby(["layout", "system"], as_index=False)["performance"].mean()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_df, x="system", y="performance", hue="layout", marker="o")
plt.title("Average Performance by Layout and System")
plt.ylabel("Performance")
plt.xlabel("System")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
