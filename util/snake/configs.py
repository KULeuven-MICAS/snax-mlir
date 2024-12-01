from util.snake.flags import get_default_flags
from util.snake.paths import get_default_paths


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
