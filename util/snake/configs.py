import os

from util.snake.flags import (
    get_clang_flags,
    get_default_flags,
    get_mlir_postproc_flags,
    get_mlir_preproc_flags,
)
from util.snake.paths import get_default_paths


def get_snax_mac_config():
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    snitch_sw_path = snax_utils_path + "/snax-mac"
    config = {}
    config.update(get_default_paths())
    config.update(get_default_flags(snitch_sw_path))
    config["vltsim"] = f"{snax_utils_path}/snax-mac-rtl/bin/snitch_cluster.vlt"
    config["cflags"].append(
        f"-I{snitch_sw_path}/target/snitch_cluster/sw/snax/mac/include"
    )
    config["ldflags"].append(
        f"-I{snitch_sw_path}/target/snitch_cluster/sw/snax/mac/build/mac.o"
    )
    return config


def get_snax_gemmx_config():
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    snitch_sw_path = snax_utils_path + "/snax-kul-cluster-mixed-narrow-wide"
    config = {}
    config.update(get_default_paths())
    config.update(get_default_flags(snitch_sw_path))
    config["vltsim"] = (
        snax_utils_path
        + "/snax-kul-cluster-mixed-narrow-wide-rtl/bin/snitch_cluster.vlt"
    )
    return config


def get_snax_alu_config():
    # use CONDA_PREFIX to access pixi env
    snax_utils_path = os.environ["CONDA_PREFIX"] + "/snax-utils"
    snitch_sw_path = snax_utils_path + "/snax-alu"
    config = {}
    config.update(get_default_paths())
    config.update(get_default_flags(snitch_sw_path))
    config["vltsim"] = snax_utils_path + "/snax-alu-rtl/bin/snitch_cluster.vlt"
    return config


def get_mlperf_tiny_config():
    config = {}
    config.update(get_default_paths())
    config.update({"clangflags": get_clang_flags()})
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
                    "clear-memory-space",
                ]
            )
        }
    )
    config.update({"mlirpostprocflags": get_mlir_postproc_flags()})
    config.update({"mlirpreprocflags": get_mlir_preproc_flags()})
    return config
