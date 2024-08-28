import os

import yaml

gemm_hardware = {
    "name": "gemm",
    "memories": {
        "reg_O": {
            "size": 32,
            "r_bw": 32,
            "w_bw": 32,
            "r_cost": 0.02,
            "w_cost": 0.02,
            "area": 0,
            "r_port": 2,
            "w_port": 2,
            "rw_port": 0,
            "latency": 1,
            "auto_cost_extraction": False,
            "operands": ["O"],
            "ports": [
                {
                    "fh": "w_port_1",
                    "tl": "r_port_1",
                    "fl": "w_port_2",
                    "th": "r_port_2",
                },
            ],
            "served_dimensions": ["D2"],
        },
        "l1": {
            "size": 262144,
            "r_bw": 2048,
            "w_bw": 2048,
            "r_cost": 22.9,
            "w_cost": 52.01,
            "area": 0,
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
            "latency": 1,
            "operands": ["I1", "I2", "O"],
            "min_r_granularity": 64,
            "min_w_granularity": 64,
            "ports": [
                {"fh": "rw_port_1", "tl": "rw_port_1"},
                {"fh": "rw_port_1", "tl": "rw_port_1"},
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1",
                    "fl": "rw_port_1",
                    "th": "rw_port_1",
                },
            ],
            "served_dimensions": ["D1", "D2", "D3"],
        },
        "l3": {
            "size": 10000000000,
            "r_bw": 512,
            "w_bw": 512,
            "r_cost": 700,
            "w_cost": 750,
            "area": 0,
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
            "latency": 1,
            "operands": ["I1", "I2", "O"],
            "ports": [
                {"fh": "rw_port_1", "tl": "rw_port_1"},
                {"fh": "rw_port_1", "tl": "rw_port_1"},
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1",
                    "fl": "rw_port_1",
                    "th": "rw_port_1",
                },
            ],
            "served_dimensions": ["D1", "D2", "D3"],
        },
    },
    "operational_array": {
        "input_precision": [8, 8],
        "multiplier_energy": 0.04,
        "multiplier_area": 1,
        "dimensions": ["D1", "D2", "D3"],
        "sizes": [8, 8, 8],
    },
}

gemm_mappping = [
    {
        "name": "default",
        "core_allocation": [1],
        "spatial_mapping": {"D1": ["B, 8"], "D2": ["C, 8"], "D3": ["K, 8"]},
        "temporal_ordering": [
            ["C", "*"],
            ["K", "*"],
            ["B", "*"],
            ["*", "*"],
        ],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    }
]


def get_yaml_files(MKN, WIO_element_type) -> dict[str, str]:
    gemm_workload = [
        {
            "id": 0,  # conv1, stride 2
            "operator_type": "Gemm",
            "equation": "O[b][k]+=I[b][c]*W[c][k]",
            "dimension_relations": [],
            "loop_dims": ["B", "C", "K"],
            "loop_sizes": [MKN[0], MKN[2], MKN[1]],
            "operand_precision": {
                "O": WIO_element_type[2],
                "O_final": WIO_element_type[1],
                "W": WIO_element_type[1],
                "I": WIO_element_type[0],
            },
            "operand_source": {"W": 0, "I": 0},
        }
    ]
    with open("gemm_hardware.yaml", "w") as file:
        yaml.dump(gemm_hardware, file, default_flow_style=False, sort_keys=False)
    with open("gemm_mapping.yaml", "w") as file:
        yaml.dump(gemm_mappping, file, default_flow_style=False, sort_keys=False)
    with open("gemm_workload.yaml", "w") as file:
        yaml.dump(gemm_workload, file, default_flow_style=False, sort_keys=False)
    return {
        "hardware": "gemm_hardware.yaml",
        "mapping": "gemm_mapping.yaml",
        "workload": "gemm_workload.yaml",
    }


def remove_yaml_files(file_paths: dict[str, str]):
    for file_path in file_paths.values():
        os.remove(file_path)
