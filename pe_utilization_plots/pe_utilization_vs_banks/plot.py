# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from matplotlib.ticker import LogFormatter, LogLocator
#
# # === Input files ===
# files = {
#     "Early\nConvolution": "earlyconv_summary.txt",
#     "Late\nConvolution": "lateconv_summary.txt",
#     "Matrix-Matrix\nMultiplication": "matmul_summary.txt",
#     "Matrix-Vector\nMultiplication": "matvec_summary.txt",
# }
#
# # === OPS per trace ===
# ops_per_trace = {
#     "earlyconv_sysmat_traces.json": 3 * 3 * 3 * 16 * 16 * 16,
#     "earlyconv_sysvec_traces.json": 3 * 3 * 3 * 16 * 16 * 16,
#     "lateconv_sysmat_traces.json": 3 * 3 * 32 * 32 * 8 * 8,
#     "lateconv_sysvec_traces.json": 3 * 3 * 32 * 32 * 8 * 8,
#     "matmul_sysmat_traces.json": 64 * 64 * 64,
#     "matmul_sysvec_traces.json": 64 * 64 * 64,
#     "matvec_sysmat_traces.json": 256 * 256,
#     "matvec_sysvec_traces.json": 256 * 256,
# }
#
# # === Load data into a DataFrame ===
# records = []
#
# for op_name, filename in files.items():
#     with open(filename) as f:
#         for line in f:
#             trace_file, cycles_str = line.strip().split("\t")
#             cycles = int(cycles_str)
#             ops = ops_per_trace[trace_file]
#             ops_per_cycle = ops / cycles
#             kind = "MatMul" if "sysmat" in trace_file else "MatVec"
#             records.append({"Operation": op_name, "System": kind, "OPS/cycle": ops_per_cycle})
#
# df = pd.DataFrame(records)
#
# # === Plotting ===
# plt.figure(figsize=(10, 6))
# sns.set(style="whitegrid")
#
# # Reverse operation order for better appearance in hbar
# df["Operation"] = pd.Categorical(df["Operation"], categories=sorted(df["Operation"].unique(), reverse=True))
#
# ax = sns.barplot(data=df, y="Operation", x="OPS/cycle", hue="System", orient="h")
#
# # Set log scale
# ax.set_xscale("log")
#
# # Major ticks: powers of 10
# ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
# ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
#
# # Minor ticks: log subdivisions like 2, 3, 5, etc.
# ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=100))
# ax.xaxis.set_minor_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
#
# # Explicit tick values (adjust based on your data range)
# tick_vals = [20, 30, 50, 100, 200, 300, 400, 500]
#
# # Set custom ticks and labels
# ax.set_xticks(tick_vals)
# ax.set_xticklabels([str(v) for v in tick_vals])
#
# # Show grid on both
# ax.grid(which="both", axis="x", linestyle="-", linewidth=1)
#
# # Optional: rotate labels if they overlap
# plt.xticks(rotation=45)
# # Log scale for x-axis
# # ax.set_xscale("log")
# # # Set major and minor ticks
# # ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
# # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))
#
# # Show grid on both
# # ax.grid(which="both", axis="x", linestyle="-", linewidth=1)
# # ax.set_title("Effective OPS/cycle: MatMul vs MatVec")
# ax.set_xlabel("Effective OPS/cycle")
# ax.set_ylabel("")
#
# # Add annotations
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.1f", label_type="center", fontsize=11, color="white")
#
# plt.legend(loc="lower right")
# plt.tight_layout()
#
# # Save to PNG
# plt.savefig("plot.pdf", dpi=300)
#

import matplotlib.pyplot as plt
import pandas as pd

# === Input files ===
files = {
    "Matrix-Matrix\nMultiplication": "matmul_summary.txt",
}

# === OPS per trace ===
expected_cycles_per_trace = {
    # "matmul_sysmat_bank1_traces.json": 5*7*9,
    # "matmul_sysmat_bank2_traces.json": 5*7*9,
    # "matmul_sysmat_bank4_traces.json": 5*7*9,
    "matmul_sysmat_archreg_bank8_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archreg_bank16_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archreg_bank32_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archreg_bank64_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archreg_bank128_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archcached_bank8_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archcached_bank16_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archcached_bank32_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archcached_bank64_traces.json": 5 * 7 * 9,
    "matmul_sysmat_archcached_bank128_traces.json": 5 * 7 * 9,
}

# === Load data into a DataFrame ===
def _detect_arch(trace_filename: str) -> str:
    if "archreg" in trace_filename:
        return "archreg"
    if "archcached" in trace_filename:
        return "archcached"
    return "unknown"

records = []
for op_name, filename in files.items():
    with open(filename) as f:
        for line in f:
            trace_file, cycles_str = line.strip().split("\t")
            cycles = int(cycles_str)
            expected_cycles = expected_cycles_per_trace[trace_file]
            utilization = expected_cycles / cycles
            kind = "MatMul" if "sysmat" in trace_file else "MatVec"
            bank_size = int(trace_file.split("_bank")[-1].split("_")[0])
            arch = _detect_arch(trace_file)
            records.append(
                {
                    "Operation": op_name,
                    "System": kind,
                    "Utilization": utilization,
                    "Bank Size": bank_size,
                    "Arch": arch,
                    "Trace": trace_file,
                }
            )

df = pd.DataFrame(records)

plt.rcParams["font.family"] = "Roboto"  # Ensure Roboto is installed on your system


# Note: Arch is assigned when building `records` from the summary file (see loop above)


# === Create a single figure with two subplots: archreg (left) and archcached (right) ===
def plot_arch_compare(df_in: pd.DataFrame, out_path: str):
    """Single overlaid plot: both architectures on one axes.

    Each architecture is plotted as a separate line (different style). If multiple
    Operations exist, lines are labelled as "<arch> - <operation>".
    """
    arches = ['archreg', 'archcached']
    styles = {'archreg': {'linestyle': '-', 'marker': 'o', 'color': 'C0'},
              'archcached': {'linestyle': '--', 'marker': 's', 'color': 'C1'}}
    
    archdict = {'archreg': 'Regular Memory Access',
                'archcached': 'Fixed Level Cache'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine x ticks as the union of bank sizes present
    bank_sizes = sorted(df_in['Bank Size'].unique())

    plotted_any = False
    for arch in arches:
        df_arch = df_in[df_in['Arch'] == arch].copy()
        if df_arch.empty:
            continue

        df_arch = df_arch.sort_values('Bank Size')

        # If multiple operations, plot each; otherwise plot overall
        for operation in df_arch['Operation'].unique():
            df_op = df_arch[df_arch['Operation'] == operation]
            label = f"{archdict.get(arch, arch)}"
            s = styles.get(arch, {})
            ax.plot(
                df_op['Bank Size'],
                df_op['Utilization'],
                marker=s.get('marker', 'o'),
                linestyle=s.get('linestyle', '-'),
                color=s.get('color', None),
                linewidth=2,
                markersize=7,
                label=label,
            )
            plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')

    ax.set_xscale('log', base=2)
    ax.set_xticks(bank_sizes)
    ax.set_xticklabels([str(int(b)) for b in bank_sizes])
    ax.grid(which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Number of Banks', fontsize=12)
    ax.set_ylabel('PE Utilization', fontsize=13)
    ax.set_ylim(bottom=0)
    ax.set_title('PE Utilization for both Architectures', fontsize=14)
    # Make legend a bit bigger and easier to read
    ax.legend(loc='best', fontsize=13, markerscale=1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# === Generate combined figure ===
plot_arch_compare(df, 'plot_arch_compare.png')
