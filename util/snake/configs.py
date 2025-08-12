import os
from pathlib import Path
from typing import Any

from util.snake.flags import (
    get_clang_flags,
    get_default_flags,
)
from util.snake.paths import get_default_paths, get_default_snax_paths


def get_snax_mac_config(snax_mlir_path: str | None = None) -> dict[str, Any]:
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    snitch_sw_path = snax_utils_path + "/snax-mac"
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update(get_default_snax_paths())
    config.update(get_default_flags(snitch_sw_path, snax_mlir_path=snax_mlir_path))
    config["num_chips"] = 1
    config["num_harts"] = 2
    config["vltsim"] = f"{snax_utils_path}/snax-mac-rtl/bin/snitch_cluster.vlt"
    config["cflags"].append(f"-I{snitch_sw_path}/target/snitch_cluster/sw/snax/mac/include")
    config["ldflags"].append(f"-I{snitch_sw_path}/target/snitch_cluster/sw/snax/mac/build/mac.o")
    return config


def get_snax_gemmx_config(snax_mlir_path: str | None = None) -> dict[str, Any]:
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    utils_dir = Path(__file__).resolve().parent.parent
    snitch_sw_path = snax_utils_path + "/snax-kul-cluster-mixed-narrow-wide"
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update(get_default_snax_paths())
    config.update(get_default_flags(snitch_sw_path, snax_mlir_path=snax_mlir_path))
    config["num_chips"] = 1
    config["num_harts"] = 2
    config["vltsim"] = snax_utils_path + "/snax-kul-cluster-mixed-narrow-wide-rtl/bin/snitch_cluster.vlt"
    config["snaxc-config"] = utils_dir / "ssot" / "snax_gemmx.yaml"
    return config


def get_snax_xdma_config(snax_mlir_path: str | None = None) -> dict[str, Any]:
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    utils_dir = Path(__file__).resolve().parent.parent
    snitch_sw_path = snax_utils_path + "/snax-kul-cluster-mixed-narrow-wide"
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update(get_default_snax_paths())
    config.update(get_default_flags(snitch_sw_path, snax_mlir_path=snax_mlir_path))
    config["num_chips"] = 1
    config["num_harts"] = 2
    # config["vltsim"] = snax_utils_path + "/snax-compiler_test_cluster-rtl/bin/snitch_cluster.vlt"
    config["vltsim"] = utils_dir / "vlt" / "snitch_cluster.vlt"
    config["snaxc-config"] = utils_dir / "ssot" / "snax_xdma.yaml"
    return config


def get_snax_alu_config(snax_mlir_path: str | None = None) -> dict[str, Any]:
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    utils_dir = Path(__file__).resolve().parent.parent
    snitch_sw_path = snax_utils_path + "/snax-alu"
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update(get_default_snax_paths())
    config.update(get_default_flags(snitch_sw_path, snax_mlir_path=snax_mlir_path))
    config["num_chips"] = 1
    config["num_harts"] = 2
    config["vltsim"] = snax_utils_path + "/snax-alu-rtl/bin/snitch_cluster.vlt"
    config["snaxc-config"] = utils_dir / "ssot" / "snax_alu.yaml"
    return config


def get_mlperf_tiny_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update({"clangflags": get_clang_flags()})
    config["num_chips"] = 1
    config["num_harts"] = 2
    config.update(
        {
            "snaxoptflags": ",".join(
                [
                    "preprocess",
                    "snax-bufferize",
                    "dispatch-kernels",
                    "set-memory-space",
                    "set-memory-layout",
                    "realize-memref-casts",
                    "reuse-memref-allocs",
                    "insert-sync-barrier",
                    "dispatch-regions",
                    "snax-copy-to-dma",
                    "memref-to-snax",
                    "snax-to-func",
                    "snax-allocate",
                    "clear-memory-space",
                    "postprocess",
                ]
            )
        }
    )
    return config


def get_host_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    config.update(get_default_paths())
    config.update({"cflags": ["-I ../../runtime/include"]})
    config.update({"clangflags": ["-O0"]})
    config.update({"num_chips": 1})
    config.update({"num_harts": 1})
    return config
