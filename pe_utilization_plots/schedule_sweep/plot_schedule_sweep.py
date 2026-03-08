#!/usr/bin/env python3
"""
Plot schedule sweep results: predicted cost vs actual cycles.

Reads schedule_sweep_combined.txt and creates a sorted bar chart
comparing cost model predictions with true simulation cycles.

Usage:
    python plot_schedule_sweep.py schedule_sweep_combined.txt output.pdf
"""

import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename: str):
    """Load the combined results TSV file."""
    schedule_indices = []
    predicted_costs = []
    actual_cycles = []

    with open(filename) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            idx = int(parts[0])
            cost = float(parts[1])
            cycles = int(parts[2])
            schedule_indices.append(idx)
            predicted_costs.append(cost)
            actual_cycles.append(cycles)

    return schedule_indices, predicted_costs, actual_cycles


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.txt output.pdf")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    schedule_indices, predicted_costs, actual_cycles = load_data(input_file)

    if not schedule_indices:
        print("No data found in input file.")
        sys.exit(1)

    # Sort all data by predicted cost (ascending)
    sorted_data = sorted(
        zip(schedule_indices, predicted_costs, actual_cycles),
        key=lambda x: x[1],
    )
    sorted_indices = [d[0] for d in sorted_data]
    sorted_costs = [d[1] for d in sorted_data]
    sorted_cycles = [d[2] for d in sorted_data]

    # Normalize: show everything relative to minimum cost for better comparison
    # Or just show absolute values side by side.
    n = len(sorted_indices)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, n * 1.2), 6))

    # Bar chart: predicted cost and actual cycles side by side
    bars1 = ax1.bar(
        x - bar_width / 2,
        sorted_costs,
        bar_width,
        label="Predicted Cost (latency model)",
        color="#4C72B0",
        alpha=0.85,
    )
    bars2 = ax1.bar(
        x + bar_width / 2,
        sorted_cycles,
        bar_width,
        label="Actual Cycles (simulation)",
        color="#DD8452",
        alpha=0.85,
    )

    # Labels
    ax1.set_xlabel("Schedule (sorted by predicted cost)", fontsize=12)
    ax1.set_ylabel("Cost / Cycles", fontsize=12)
    ax1.set_title(
        "Schedule Sweep: Predicted Cost vs Actual Cycles",
        fontsize=14,
        fontweight="bold",
    )

    # X tick labels: schedule index
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"sched {i}" for i in sorted_indices],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Plot saved to {output_file} ({n} schedules)",
        file=sys.stderr,
    )

    # Also print a summary table
    print("\nSchedule Sweep Summary (sorted by predicted cost):", file=sys.stderr)
    print(f"{'Sched':>7} {'Pred. Cost':>12} {'Actual Cyc.':>12} {'Ratio':>8}", file=sys.stderr)
    print("-" * 43, file=sys.stderr)
    for idx, cost, cycles in sorted_data:
        ratio = cycles / cost if cost > 0 else float("inf")
        print(f"{idx:>7} {cost:>12.1f} {cycles:>12} {ratio:>8.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
