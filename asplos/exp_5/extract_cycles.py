import json
import os
import sys

import yaml
from dacite import from_dict

from asplos.util.convspecs import TiledConfig, TiledConvLayer

if len(sys.argv) < 3:
    print("Usage: python extract_cycles.py output_file.txt input1.json input2.json ...")
    sys.exit(1)

output_file = sys.argv[1]
input_files = sys.argv[2:]

with open("tiled_resnet_layers.yaml") as f:
    yaml_data = yaml.safe_load(f)

tiled_config = from_dict(data_class=TiledConfig, data=yaml_data)

config_dict: dict[str, TiledConvLayer] = {layer.layer.name + "_traces.json": layer for layer in tiled_config.layers}


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
        config = config_dict[filename]
        out.write(f"{filename}\t{cycles}\t{config.layer.total_ops / 256 / cycles}\n")
