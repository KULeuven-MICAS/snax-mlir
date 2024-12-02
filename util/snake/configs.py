from util.snake.flags import get_clang_flags, get_default_flags, get_mlir_postproc_flags
from util.snake.paths import get_default_paths


def get_snax_alu_config():
    config = {}
    config.update(get_default_paths())
    config.update(get_default_flags("/opt/snax-alu"))
    config["vltsim"] = "/opt/snax-alu-rtl/bin/snitch_cluster.vlt"
    return config


def get_snax_mac_config():
    config = {}
    SNITCH_SW_PATH = "/opt/snax-mac"
    config.update(get_default_paths())
    config.update(get_default_flags(SNITCH_SW_PATH))
    config["vltsim"] = "/opt/snax-mac-rtl/bin/snitch_cluster.vlt"
    config["cflags"].append(
        f"-I{SNITCH_SW_PATH}/target/snitch_cluster/sw/snax/mac/include"
    )
    config["ldflags"].append(
        f"-I{SNITCH_SW_PATH}/target/snitch_cluster/sw/snax/mac/build/mac.o"
    )
    return config


def get_snax_gemmx_config():
    config = {}
    SNITCH_SW_PATH = "/opt/snax-kul-cluster-mixed-narrow-wide/"
    config.update(get_default_paths())
    config.update(get_default_flags(SNITCH_SW_PATH))
    config[
        "vltsim"
    ] = "/opt/snax-kul-cluster-mixed-narrow-wide-rtl/bin/snitch_cluster.vlt"
    return config


def get_mlperf_tiny_config():
    config = {}
    config.update(get_default_paths())
    config.update({"clangflags": get_clang_flags()})
    config.update(
        {
            "snaxoptflags": ",".join(
                [
                    "dispatch-kernels",
                    "set-memory-space",
                    "set-memory-layout",
                    "realize-memref-casts",
                    "reuse-memref-allocs",
                    "insert-sync-barrier",
                    "dispatch-regions",
                    "linalg-to-library-call",
                    "snax-copy-to-dma",
                    "memref-to-snax",
                    "snax-to-func",
                    "clear-memory-space",
                ]
            )
        }
    )
    config.update({"mlirpostprocflags": get_mlir_postproc_flags()})
    return config
