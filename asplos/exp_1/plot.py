import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatter, LogLocator

# === Input files ===
files = {
    "Early\nConvolution": "earlyconv_summary.txt",
    "Late\nConvolution": "lateconv_summary.txt",
    "Matrix-Matrix\nMultiplication": "matmul_summary.txt",
    "Matrix-Vector\nMultiplication": "matvec_summary.txt",
}

# === OPS per trace ===
ops_per_trace = {
    "earlyconv_sysmat_traces.json": 3 * 3 * 3 * 16 * 16 * 16,
    "earlyconv_sysvec_traces.json": 3 * 3 * 3 * 16 * 16 * 16,
    "lateconv_sysmat_traces.json": 3 * 3 * 32 * 32 * 8 * 8,
    "lateconv_sysvec_traces.json": 3 * 3 * 32 * 32 * 8 * 8,
    "matmul_sysmat_traces.json": 64 * 64 * 64,
    "matmul_sysvec_traces.json": 64 * 64 * 64,
    "matvec_sysmat_traces.json": 256 * 256,
    "matvec_sysvec_traces.json": 256 * 256,
}

# === Load data into a DataFrame ===
records = []

for op_name, filename in files.items():
    with open(filename) as f:
        for line in f:
            trace_file, cycles_str = line.strip().split("\t")
            cycles = int(cycles_str)
            ops = ops_per_trace[trace_file]
            ops_per_cycle = ops / cycles
            kind = "MatMul" if "sysmat" in trace_file else "MatVec"
            records.append({"Operation": op_name, "System": kind, "OPS/cycle": ops_per_cycle})

df = pd.DataFrame(records)

# === Plotting ===
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Reverse operation order for better appearance in hbar
df["Operation"] = pd.Categorical(df["Operation"], categories=sorted(df["Operation"].unique(), reverse=True))

ax = sns.barplot(data=df, y="Operation", x="OPS/cycle", hue="System", orient="h")

# Set log scale
ax.set_xscale("log")

# Major ticks: powers of 10
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))

# Minor ticks: log subdivisions like 2, 3, 5, etc.
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=100))
ax.xaxis.set_minor_formatter(LogFormatter(base=10.0, labelOnlyBase=False))

# Explicit tick values (adjust based on your data range)
tick_vals = [20, 30, 50, 100, 200, 300, 400, 500]

# Set custom ticks and labels
ax.set_xticks(tick_vals)
ax.set_xticklabels([str(v) for v in tick_vals])

# Show grid on both
ax.grid(which="both", axis="x", linestyle="-", linewidth=1)

# Optional: rotate labels if they overlap
plt.xticks(rotation=45)
# Log scale for x-axis
# ax.set_xscale("log")
# # Set major and minor ticks
# ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
# ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))

# Show grid on both
# ax.grid(which="both", axis="x", linestyle="-", linewidth=1)
# ax.set_title("Effective OPS/cycle: MatMul vs MatVec")
ax.set_xlabel("Effective OPS/cycle")
ax.set_ylabel("")

# Add annotations
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="center", fontsize=11, color="white")

plt.legend(loc="lower right")
plt.tight_layout()

# Save to PNG
plt.savefig("plot.pdf", dpi=300)
