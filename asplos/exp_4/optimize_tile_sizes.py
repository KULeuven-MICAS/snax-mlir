from dataclasses import asdict

import yaml
from dacite import from_dict

from asplos.util.convspecs import ModelConfig, TiledConfig, TiledConvLayer

MEM_SIZE = 248_000

with open("resnet_layers.yaml") as f:
    yaml_data = yaml.safe_load(f)

model_config = from_dict(data_class=ModelConfig, data=yaml_data)


results: list[tuple[float, float, float]] = []
values = [i / 10 for i in range(1, 10)]  # 0.0 to 1.0 in steps of 0.1

for a in values:
    for b in values:
        for c in values:
            if round(a + b + c, 1) == 1.0:
                results.append((a, b, c))

for input_factor, weight_factor, output_factor in results:
    tiled_config = TiledConfig([])
    per_operand_tiled_config = TiledConfig([])
    for layer in model_config.layers:
        specs: dict[TiledConvLayer, int] = {}
        per_operand_specs: dict[TiledConvLayer, int] = {}
        for tile_k in layer.tiling_sizes_k():
            for tile_y in layer.tiling_sizes_oy():
                spec = TiledConvLayer(layer=layer, tile_k=tile_k, tile_y=tile_y)
                # filter based on the ones that fit in L1 memory
                if (  # per operand
                    spec.input_tile_size() < MEM_SIZE * input_factor
                    and spec.weight_tile_size() < MEM_SIZE * weight_factor
                    and spec.output_tile_size() < MEM_SIZE * output_factor
                ):
                    per_operand_specs[spec] = spec.total_transfer_size()
                if spec.total_tile_size() < MEM_SIZE:  # dynamic memory
                    specs[spec] = spec.total_transfer_size()
        # select the one with least amount of data transfer
        if len(specs) == 0:
            break
        tiled_config.layers.append(min(specs, key=lambda k: specs[k]))
        per_operand_tiled_config.layers.append(min(per_operand_specs, key=lambda k: per_operand_specs[k]))

    if len(tiled_config.layers) < len(model_config.layers):
        continue

    # print("Total DRAM Transfer Size (Dynamic Memory):")
    # print(
    #     sum(layer.total_transfer_size() for layer in tiled_config.layers) / 1024 / 1024,
    #     "MiB",
    # )

    # print("Total DRAM Transfer Size (Per-Operand Memory):")
    print(input_factor, weight_factor, output_factor)
    print(
        sum(layer.total_transfer_size() for layer in per_operand_tiled_config.layers) / 1024 / 1024,
        "MiB",
    )

# output to yaml file
with open("tiled_resnet_layers.yaml", "w") as f:
    yaml.dump(asdict(tiled_config), f, default_flow_style=False)
with open("per_operand_tiled_resnet_layers.yaml", "w") as f:
    yaml.dump(asdict(per_operand_tiled_config), f, default_flow_style=False)
