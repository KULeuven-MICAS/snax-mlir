import json
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python extract_cycles.py output_file.txt input1.json input2.json ...")
    sys.exit(1)

output_file = sys.argv[1]
input_files = sys.argv[2:]

with open(output_file, "w") as out:
    for infile in input_files:
        try:
            with open(infile) as f:
                data = json.load(f)
            cycles = data[0][0][1]["cycles"]
        except Exception as e:
            print(f"Error processing {infile}: {e}")
            cycles = "ERROR"
        filename = os.path.basename(infile)
        out.write(f"{filename}\t{cycles}\n")
