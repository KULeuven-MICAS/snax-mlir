import os
import json
import pandas as pd

# Define the base directory for test folders
base_dir = "results"
conv_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("conv_")]

data = []

# Process each directory
for dir_path in conv_dirs:
    # Initialize entry
    entry = {"Directory": dir_path, "Errors": None, "Cycles": None}
    
    # Check sim_generated.log for errors
    sim_log_file = os.path.join(dir_path, "sim_generated.log")
    if os.path.isfile(sim_log_file):
        try:
            with open(sim_log_file, 'r') as log_file:
                for line in log_file:
                    if "Finished, nb errors:" in line:
                        entry["Errors"] = int(line.split(":")[-1].strip())
                        break
        except Exception as e:
            entry["Errors"] = "Error reading log"

    # Check trace file for cycles
    trace_file = os.path.join(dir_path, "logs/trace_chip_00_hart_00000.trace.json")
    if os.path.isfile(trace_file):
        try:
            with open(trace_file, 'r') as trace_file_content:
                trace_data = json.load(trace_file_content)
                if len(trace_data) > 2 and 'cycles' in trace_data[2]:
                    entry["Cycles"] = trace_data[2]['cycles']
        except Exception as e:
            entry["Cycles"] = "Error reading trace"

    data.append(entry)

# Convert to DataFrame for better readability and export
df = pd.DataFrame(data)

# Save or display the results
df.to_csv("test_results_summary.csv", index=False)
print(df)

