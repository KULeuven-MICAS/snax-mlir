import matplotlib.pyplot as plt
import numpy as np

# Data preparation
categories = ["Data Movement", "Conv2D", "MaxPool", "Fully Connected"]
labels = ['CPU Execution', 'GeMM Accelerator', 'GeMM + MaxPool Accelerator', 'Pipelined Execution']
data = np.array([
    [1496, 8568894, 47368, 2413],
    [1753, 1924, 50348, 2403],
    [1598, 2017, 2173, 2408],
    [239, 2056, 1928, 2384]
])

# Normalize data for the 100% stacked bar
data_normalized = data / data.sum(axis=1, keepdims=True) * 100

# Plotting a flipped (horizontal) version with reversed order of labels
fig, ax = plt.subplots(figsize=(10, 6))

# Reverse the labels and data for plotting
reversed_labels = labels[::-1]
reversed_data_normalized = data_normalized[::-1]

# Horizontal stacked bars for first three configurations in reversed order
left = np.zeros(len(reversed_data_normalized))
for i in range(len(categories)):
    ax.barh(reversed_labels, reversed_data_normalized[:, i], left=left, label=categories[i])
    left += reversed_data_normalized[:, i]

# Pipelined Execution as a horizontal stacked bar in reversed order
# y_pos = 0  # Now at the top position
# left_pipelined = 0
# for i in range(len(categories)):
#     ax.barh(reversed_labels[y_pos], reversed_pipelined_data[i], left=left_pipelined, label=categories[i] if i == 0 else "")
#     left_pipelined += reversed_pipelined_data[i]

# Add labels and legend
ax.set_xlabel("Percentage")
ax.set_title("100% Stacked Bar Plot of Execution Data (Horizontal, Reversed Order)")
ax.legend(title="Operations", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('plot.pdf')
