import json

import pandas as pd

# Open the test_results.json file
with open("test_results.json") as file:
    data = json.load(file)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=["layout", "backend", "size", "success", "cycles"])


def calculate_expected(size):
    sizes = size.split("x")
    sizes = [round(int(x) / 8) for x in sizes]
    return sizes[0] * sizes[1] * sizes[2]


# Add a new column to the DataFrame with the expected number of cycles
df["expected"] = df["size"].apply(calculate_expected)

# add a new column to the DataFrame with the speedup
df["utilization"] = df["expected"] / df["cycles"]

# order the data by layout, backend, and size
df = df.sort_values(by=["success", "layout", "backend", "size"])
df.to_csv("summary.csv")

print(df.head(30))

# Calculate the success rate
success_rate = df["success"].sum() / len(df)
print("Success Rate:", success_rate)
