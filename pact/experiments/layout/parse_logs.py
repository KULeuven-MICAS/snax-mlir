import json
import os

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

dirs = [
    d
    for d in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, d)) and d.startswith("test_")
]

results = []

for d in dirs:
    file_to_open = os.path.join(
        directory, d, "matmul.x.logs", "trace_hart_00000000.trace.json"
    )
    with open(file_to_open) as file:
        file = json.load(file)
        result = file[2]
        name_split = d.split("_")
        layout = name_split[2]
        backend = name_split[3]
        size = "x".join(name_split[4:])
        result["test"] = d
        result["layout"] = layout
        result["backend"] = backend
        result["size"] = size
        result["success"] = len(file) == 5
        results.append(result)

# Store results in results.json
with open("test_results.json", "w") as outfile:
    json.dump(results, outfile)
