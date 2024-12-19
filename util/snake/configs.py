import os

from util.snake.flags import get_default_flags
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
