from dataclasses import asdict

import yaml
from dacite import from_dict

from asplos.util.convspecs import ModelConfig, TiledConfig, TiledConvLayer

MEM_SIZE = 248_000

with open("resnet_layers.yaml") as f:
    yaml_data = yaml.safe_load(f)

model_config = from_dict(data_class=ModelConfig, data=yaml_data)

tiled_config = TiledConfig([])

for layer in model_config.layers:
    specs: dict[TiledConvLayer, int] = {}
    for tile_k in layer.tiling_sizes_k():
        for tile_y in layer.tiling_sizes_oy():
            spec = TiledConvLayer(layer=layer, tile_k=tile_k, tile_y=tile_y)
            # filter based on the ones that fit in L1 memory
            if spec.total_tile_size() < MEM_SIZE:
                specs[spec] = spec.total_transfer_size()
    # select the one with least amount of data transfer
    tiled_config.layers.append(min(specs, key=lambda k: specs[k]))

# max_weight_size = max(layer.weight_tile_size() for layer in tiled_config.layers)
# max_input_size = max(layer.input_tile_size() for layer in tiled_config.layers)
# max_output_size = max(layer.output_tile_size() for layer in tiled_config.layers)
# max_added_size = max_weight_size + max_input_size + max_output_size
# max_total_size = max(layer.total_tile_size() for layer in tiled_config.layers)
#
# print(f"Max weight tile size: {max_weight_size}")
# print(f"Max input tile size: {max_input_size}")
# print(f"Max output tile size: {max_output_size}")
# print(f"Sum of three: {max_added_size}")
# print(f"Max total tile size with dynamic memory: {max_total_size} ({max_total_size / max_added_size:2.2%})")

# output to yaml file
with open("tiled_resnet_layers.yaml", "w") as f:
    yaml.dump(asdict(tiled_config), f, default_flow_style=False)
