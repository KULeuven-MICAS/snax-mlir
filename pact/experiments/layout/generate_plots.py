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

colors = [
    "#e03131",
    "#1971c2",
    "#f08c00",
    "#9c36b5",
    "#2f9e44",
    "#f0c800",
    "#f00000",
    "#00f000",
    "#0000f0",
    "#f000f0",
]
color_background = "#ebebeb"


def plot_experiment_1a(df):
    # Filter the dataframe
    filtered_df = df[
        (df["backend"] == "fifo-1-slow")
        & (df["layout"].isin(["default", "tiled", "strided", "round-robin"]))
        & (df["success"] is True)
    ]

    # Sort values by 'ops' and 'utilization'
    sorted_df = filtered_df.sort_values(["ops", "utilization"])

    # Plotting
    _, ax = plt.subplots(ncols=2, width_ratios=[7, 2], figsize=(14, 5))
    for layout, color in zip(
        ["default", "strided", "round-robin", "tiled"], colors[0:4]
    ):
        layout_df = sorted_df[sorted_df["layout"] == layout]
        # Use numpy.arange for x values to ensure equidistance
        x_values = range(len(layout_df))
        ax[0].plot(
            x_values,
            layout_df["utilization"],
            marker=".",
            linestyle="--",
            linewidth=0.51,
            label=f"{layout} layout",
            color=color,
        )

    # Set labels
    x_labels = sorted_df[sorted_df["layout"] == "default"]
    x_labels = x_labels["ops"]
    x_labels = x_labels.apply(lambda x: f"{round(x/1e3)}k")
    x_labels = x_labels[::20]
    ax[0].set_xticks(range(0, 260, 20))
    ax[0].set_xticklabels(x_labels, fontproperties=labels_font)

    # Set y-axis limits
    ax[0].set_ylim(0, 1)
    # Set y tick label font
    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontproperties=labels_font)

    # Set axis titles font
    ax[0].set_xlabel("Operation MACs", fontproperties=axis_font)
    ax[0].set_ylabel("Utilization", fontproperties=axis_font)

    # Enable white grid and set light blue background
    ax[0].grid(True, color="white")
    ax[0].set_facecolor(color_background)

    # Set the color for the legend
    legend = ax[0].legend(
        labels=["non-strided", "strided", "tiled", "tiled + banked"], prop=legend_font
    )
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)

    # create bottom figure - a violin plot
    violin_data = []
    for layout in ["default", "strided", "round-robin", "tiled"]:
        violin_data.append(
            sorted_df[sorted_df["layout"] == layout]["utilization"].values
        )

    violin_parts = ax[1].violinplot(
        dataset=violin_data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        vert=True,
        widths=0.6,
        points=100,
    )

    for i, median in enumerate(violin_parts["cmedians"].get_paths()):
        median_x = median.vertices[:, 0]
        median_y = median.vertices[:, 1]
        if i <= 1:
            xytext = (3, 1)
            ha = "right"
            va = "bottom"
        else:
            xytext = (8, 1)
            ha = "left"
            va = "bottom"
        layouthere = ["default", "strided", "round-robin", "tiled"]
        layouthere = layouthere[i]
        annotations = sorted_df[sorted_df["layout"] == layouthere]["utilization"]
        annotations = annotations.median()
        ax[1].annotate(
            f"{annotations:.2f}",
            xy=(median_x[len(median_x) // 2], median_y[len(median_y) // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colors[i],
            fontproperties=labels_font_small,
        )

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    violin_parts["cmedians"].set_colors(colors[0:4])
    violin_parts["cmaxes"].set_colors(colors[0:4])
    violin_parts["cmins"].set_colors(colors[0:4])
    violin_parts["cbars"].set_colors(colors[0:4])

    # reverse y axis
    ax[1].invert_xaxis()
    # set y axis labels
    # ax[1].set_yticks(range(1, 5))
    # disable y axis
    ax[1].set_xticks([])

    ax[1].grid(True, color="white")
    ax[1].set_facecolor(color_background)

    # print(ax[1].get_xlim())

    ax[1].set_ylabel("Average Utilization", fontproperties=axis_font)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "plots", "experiment_1a.png"), dpi=300)


def plot_experiment_1b(df):
    # Filter the dataframe
    filtered_df = df[
        (df["backend"] == "fifo-1-slow")
        & (df["layout"].isin(["default", "strided"]))
        & (df["success"] is True)
    ]

    layouts = ["default", "strided"]
    layout_map = {"default": "non-strided", "strided": "strided"}

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for i, layout in enumerate(layouts):
        layout_df = filtered_df[filtered_df["layout"] == layout]
        data = layout_df.groupby(["K", "N"])["utilization"].mean()
        array_data = data.unstack()
        index = data.unstack().index

        # plot a heatmap with the array_data
        axs[i].imshow(array_data, cmap="cividis", interpolation="nearest")
        axs[i].set_xticks(range(len(index)))
        axs[i].set_yticks(range(len(index)))
        axs[i].set_xticklabels(
            index, fontproperties=labels_font_small
        )  # Set x-axis tick labels font
        axs[i].set_yticklabels(
            index, fontproperties=labels_font_small
        )  # Set y-axis tick labels font
        axs[i].set_xlabel("K", fontproperties=labels_font)  # Set x-axis label font
        axs[i].set_ylabel("N", fontproperties=labels_font)  # Set y-axis label font
        axs[i].set_title(
            f"Utilization for the {layout_map[layout]} layout", fontproperties=axis_font
        )

        # Add annotations
        for j in range(len(index)):
            for k in range(len(index)):
                axs[i].text(
                    k,
                    j,
                    f"{array_data.iloc[j, k]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontproperties=labels_font_small,
                )

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "plots", "experiment_1b.png"), dpi=300)
    # plt.show()


def plot_experiment_2a(df):
    # Filter the dataframe
    filtered_df = df[
        (df["layout"].isin(["default", "tiled"])) & (df["success"] is True)
    ]

    # Sort values by 'ops' and 'utilization'
    sorted_df = filtered_df.sort_values(["ops", "utilization"])

    # Plotting
    _, ax = plt.subplots(ncols=2, width_ratios=[6, 1], figsize=(8, 6))

    thiscolors = [colors[0]] * 5 + [colors[3]]

    # create bottom figure - a violin plot
    violin_data = []
    for backend in ["fifo-1-slow", "fifo-1", "fifo-2", "fifo-3", "fifo-4"]:
        violin_data.append(
            sorted_df[
                (sorted_df["backend"] == backend) & (sorted_df["layout"] == "default")
            ]["utilization"].values
        )

    # violin_data.append(sorted_df[
    #     (sorted_df["backend"] == "fifo-1-slow") &
    #     (sorted_df["layout"] == "tiled")]['utilization'].values)

    violin_parts = ax[0].violinplot(
        dataset=violin_data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        vert=True,
        widths=0.6,
        points=100,
    )

    # for i, pc in enumerate(violin_parts["bodies"]):
    #     # Calculate the x position for the annotation
    #     x = max(violin_data[i]) + 0.05

    #     # Calculate the y position for the annotation
    #     y = i + 1

    #     # Add the annotation

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(thiscolors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    violin_parts["cmedians"].set_colors(thiscolors[0:6])
    violin_parts["cmaxes"].set_colors(thiscolors[0:6])
    violin_parts["cmins"].set_colors(thiscolors[0:6])
    violin_parts["cbars"].set_colors(thiscolors[0:6])

    # reverse y axis
    # ax[0].invert_xaxis()
    # set y axis labels
    ax[0].set_xticks(range(1, 7))
    ax[0].set_xticklabels(["1", "2", "3", "4", "5", ""], fontproperties=labels_font)
    # disable y axis
    # ax[0].set_yticks([])
    ax[0].set_xlabel("Buffer Size", fontproperties=axis_font)

    # Create legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="non-strided",
            markerfacecolor=colors[0],
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="tiled+banked",
            markerfacecolor=colors[3],
            markersize=8,
        ),
    ]
    ax[0].legend(handles=legend_elements, loc="upper left", prop=legend_font)

    ax[0].grid(True, color="white")
    ax[0].set_facecolor(color_background)

    ax[0].set_ylabel("Average Utilization", fontproperties=axis_font)

    for i, median in enumerate(violin_parts["cmedians"].get_paths()):
        median_x = median.vertices[:, 0]
        median_y = median.vertices[:, 1]
        xytext = (10, 0)
        ha = "left"
        va = "center"
        # layouthere = ["default", "strided", "round-robin", "tiled"]
        # layouthere = layouthere[i]
        ax[0].annotate(
            f"{median.vertices[0][1]:.2f}",
            xy=(median_x[len(median_x) // 2], median_y[len(median_y) // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colors[0],
            fontproperties=labels_font_small,
        )

        # create bottom figure - a violin plot
    violin_data = []
    # for backend in ["fifo-1-slow", "fifo-1", "fifo-2", "fifo-3", "fifo-4"]:
    #     violin_data.append(sorted_df[
    #         (sorted_df["backend"] == backend) &
    #         (sorted_df["layout"] == "default")]['utilization'].values)

    violin_data.append(
        sorted_df[
            (sorted_df["backend"] == "fifo-1-slow") & (sorted_df["layout"] == "tiled")
        ]["utilization"].values
    )

    violin_parts = ax[1].violinplot(
        dataset=violin_data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        vert=True,
        widths=0.3,
        points=100,
    )

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[3])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    for i, median in enumerate(violin_parts["cmedians"].get_paths()):
        median_x = median.vertices[:, 0]
        median_y = median.vertices[:, 1]
        xytext = (-6, 0)
        ha = "left"
        va = "bottom"
        # layouthere = ["default", "strided", "round-robin", "tiled"]
        # layouthere = layouthere[i]
        ax[1].annotate(
            f"{median.vertices[0][1]:.2f}",
            xy=(median_x[len(median_x) // 2], median_y[len(median_y) // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colors[3],
            fontproperties=labels_font_small,
        )

    violin_parts["cmedians"].set_color(colors[3])
    violin_parts["cmaxes"].set_color(colors[3])
    violin_parts["cmins"].set_color(colors[3])
    violin_parts["cbars"].set_color(colors[3])

    # reverse y axis
    # ax[0].invert_yaxis()
    # set y axis labels
    # ax[0].set_yticks(range(1, 7))
    # disable y axis
    ax[1].set_xticks([])
    # ax[0].set_ylabel("Buffer Size", fontproperties=axis_font)

    # # Create legend
    # legend_elements = [
    # ]
    # ax[1].legend(handles=legend_elements, loc='upper left', prop=legend_font)

    ax[1].grid(True, color="white")
    ax[1].set_facecolor(color_background)

    # ax[1].set_ylabel("Average Utilization", fontproperties=axis_font)
    # remove y labels
    ax[1].set_yticklabels([""] * len(ax[1].get_yticks()))

    ax[1].set_ylim(ax[0].get_ylim())

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "plots", "experiment_2a.png"), dpi=300)


def plot_experiment_3a(df):
    # Filter the dataframe
    filtered_df = df[
        (df["layout"].isin(["round-robin", "tiled"])) & (df["success"] is True)
    ]

    # Sort values by 'ops' and 'utilization'
    sorted_df = filtered_df.sort_values(["ops", "utilization"])

    # Plotting
    _, ax = plt.subplots(ncols=2, width_ratios=[6, 1], figsize=(8, 6))

    thiscolors = [colors[4]] * 5 + [colors[3]]

    # create bottom figure - a violin plot
    violin_data = []
    for backend in ["fifo-1-slow", "fifo-1", "fifo-2", "fifo-3", "fifo-4"]:
        violin_data.append(
            sorted_df[
                (sorted_df["backend"] == backend)
                & (sorted_df["layout"] == "round-robin")
            ]["utilization"].values
        )

    # violin_data.append(sorted_df[
    #     (sorted_df["backend"] == "fifo-1-slow") &
    #     (sorted_df["layout"] == "tiled")]['utilization'].values)

    violin_parts = ax[0].violinplot(
        dataset=violin_data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        vert=True,
        widths=0.6,
        points=100,
    )

    # for i, pc in enumerate(violin_parts["bodies"]):
    #     # Calculate the x position for the annotation
    #     x = max(violin_data[i]) + 0.05

    #     # Calculate the y position for the annotation
    #     y = i + 1

    #     # Add the annotation

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(thiscolors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    violin_parts["cmedians"].set_colors(thiscolors[0:6])
    violin_parts["cmaxes"].set_colors(thiscolors[0:6])
    violin_parts["cmins"].set_colors(thiscolors[0:6])
    violin_parts["cbars"].set_colors(thiscolors[0:6])

    # reverse y axis
    # ax[0].invert_xaxis()
    # set y axis labels
    ax[0].set_xticks(range(1, 7))
    ax[0].set_xticklabels(["1", "2", "3", "4", "5", ""], fontproperties=labels_font)
    # disable y axis
    # ax[0].set_yticks([])
    ax[0].set_xlabel("Buffer Size", fontproperties=axis_font)

    # Create legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="tiled + cycling access",
            markerfacecolor=colors[4],
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="tiled+banked",
            markerfacecolor=colors[3],
            markersize=8,
        ),
    ]
    ax[0].legend(handles=legend_elements, loc="upper left", prop=legend_font)

    ax[0].grid(True, color="white")
    ax[0].set_facecolor(color_background)

    ax[0].set_ylabel("Average Utilization", fontproperties=axis_font)

    for i, median in enumerate(violin_parts["cmedians"].get_paths()):
        median_x = median.vertices[:, 0]
        median_y = median.vertices[:, 1]
        xytext = (10, 0)
        ha = "left"
        va = "center"
        # layouthere = ["default", "strided", "round-robin", "tiled"]
        # layouthere = layouthere[i]
        ax[0].annotate(
            f"{median.vertices[0][1]:.2f}",
            xy=(median_x[len(median_x) // 2], median_y[len(median_y) // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colors[4],
            fontproperties=labels_font_small,
        )

        # create bottom figure - a violin plot
    violin_data = []
    # for backend in ["fifo-1-slow", "fifo-1", "fifo-2", "fifo-3", "fifo-4"]:
    #     violin_data.append(sorted_df[
    #         (sorted_df["backend"] == backend) &
    #         (sorted_df["layout"] == "default")]['utilization'].values)

    violin_data.append(
        sorted_df[
            (sorted_df["backend"] == "fifo-1-slow") & (sorted_df["layout"] == "tiled")
        ]["utilization"].values
    )

    violin_parts = ax[1].violinplot(
        dataset=violin_data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        vert=True,
        widths=0.3,
        points=100,
    )

    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[3])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    for i, median in enumerate(violin_parts["cmedians"].get_paths()):
        median_x = median.vertices[:, 0]
        median_y = median.vertices[:, 1]
        xytext = (-6, 0)
        ha = "left"
        va = "bottom"
        # layouthere = ["default", "strided", "round-robin", "tiled"]
        # layouthere = layouthere[i]
        ax[1].annotate(
            f"{median.vertices[0][1]:.2f}",
            xy=(median_x[len(median_x) // 2], median_y[len(median_y) // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colors[3],
            fontproperties=labels_font_small,
        )

    violin_parts["cmedians"].set_color(colors[3])
    violin_parts["cmaxes"].set_color(colors[3])
    violin_parts["cmins"].set_color(colors[3])
    violin_parts["cbars"].set_color(colors[3])

    # reverse y axis
    # ax[0].invert_yaxis()
    # set y axis labels
    # ax[0].set_yticks(range(1, 7))
    # disable y axis
    ax[1].set_xticks([])
    # ax[0].set_ylabel("Buffer Size", fontproperties=axis_font)

    # # Create legend
    # legend_elements = [    # ]
    # ax[1].legend(handles=legend_elements, loc='upper left', prop=legend_font)

    ax[1].grid(True, color="white")
    ax[1].set_facecolor(color_background)

    # ax[1].set_ylabel("Average Utilization", fontproperties=axis_font)
    # remove y labels
    ax[1].set_yticklabels([""] * len(ax[1].get_yticks()))

    ax[1].set_ylim(ax[0].get_ylim())

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "plots", "experiment_3a.png"), dpi=300)


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv(os.path.join(directory, "summary.csv"))

    plot_experiment_1a(data)
    plot_experiment_1b(data)
    plot_experiment_2a(data)
    plot_experiment_3a(data)
