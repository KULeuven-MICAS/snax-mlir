import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# get directory of current file
directory = os.path.dirname(__file__)

legend_font = fm.FontProperties(family="serif", weight="bold")
labels_font = fm.FontProperties(family="serif")
labels_font_small = fm.FontProperties(family="serif", size=8)
axis_font = fm.FontProperties(family="serif")

colors = ["#e03131", "#1971c2", "#f08c00", "#9c36b5", "#2f9e44"]


def plot_experiment_1a(df):
    # Filter the dataframe
    filtered_df = df[
        (df["backend"] == "fifo-1-slow") & (df["layout"].isin(["default", "tiled"]))
    ]

    # Sort values by 'ops' and 'utilization'
    sorted_df = filtered_df.sort_values(["ops", "utilization"])

    # Plotting
    _, ax = plt.subplots()
    for layout, color in zip(["default", "tiled"], ["#e03131", "#1971c2"]):
        layout_df = sorted_df[sorted_df["layout"] == layout]
        # Use numpy.arange for x values to ensure equidistance
        x_values = range(len(layout_df))
        ax.plot(
            x_values,
            layout_df["utilization"],
            marker="o",
            label=f"{layout} layout",
            color=color,
        )

    # Create legend
    ax.legend(
        frameon=False, prop=legend_font
    )  # Remove border from the legend and set serif font

    # Set labels
    x_labels = sorted_df[sorted_df["layout"] == "default"]
    x_labels = x_labels["size"]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(
        x_labels, rotation=55, ha="right", fontproperties=labels_font_small
    )

    # Set y-axis limits
    ax.set_ylim(0, 1)
    # Set y tick label font
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=labels_font)

    # Set axis titles font
    ax.set_xlabel("Operation Size", fontproperties=axis_font)
    ax.set_ylabel("Utilization", fontproperties=axis_font)

    # Enable white grid and set light blue background
    ax.grid(True, color="white")
    ax.set_facecolor("#E6F0F5")

    # Set the color for the legend
    legend = ax.legend(frameon=False, prop=legend_font)
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "plots", "experiment_1a.png"), dpi=300)


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv(os.path.join(directory, "summary.csv"))

    plot_experiment_1a(data)
