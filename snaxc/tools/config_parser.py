import re
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import StringAttr
from xdsl.utils.hints import isa

from snaxc.accelerators.acc_context import AccContext
from snaxc.accelerators.snax_alu import SNAXAluAccelerator
from snaxc.accelerators.snax_gemmx import SNAXGEMMXAccelerator
from snaxc.accelerators.snax_xdma import SNAXXDMAAccelerator
from snaxc.util.snax_memory import SnaxMemory


@dataclass
class Core:
    accelerators: list[str]


@dataclass
class Cluster:
    memory: SnaxMemory
    cores: list[Core]


def parse_config(config: Any) -> AccContext:
    context = AccContext()
    assert isa(config, dict[str, Any])

    for key, value in config.items():
        if key == "memory":
            assert isa(value, dict[str, Any])
            memory = parse_memory(value)
            context.register_memory(memory)
        elif re.fullmatch(r"cluster_(\d+)", key):
            assert isa(value, dict[str, Any])
            cluster = parse_cluster(value)
            context.register_memory(cluster.memory)
            for core in cluster.cores:
                if core.accelerators:
                    for accelerator in core.accelerators:
                        if accelerator == "snax_gemmx":
                            context.register_accelerator(SNAXGEMMXAccelerator.name, lambda: SNAXGEMMXAccelerator())
                        elif accelerator == "snax_alu":
                            context.register_accelerator(SNAXAluAccelerator.name, lambda: SNAXAluAccelerator())
                        elif accelerator == "xdma":
                            context.register_accelerator(SNAXXDMAAccelerator.name, lambda: SNAXXDMAAccelerator())
        else:
            raise ValueError(f"Unknown config key: {key}")

    return context


def parse_memory(config: dict[str, Any]) -> SnaxMemory:
    assert set(config.keys()) == set(["name", "start", "size"])
    return SnaxMemory(StringAttr(config["name"]), config["size"], config["start"])


def parse_cluster(config: dict[str, Any]) -> Cluster:
    cores = [core_name for core_name in config.keys() if re.fullmatch(r"core_(\d+)", core_name)]
    return Cluster(
        memory=parse_memory(config["memory"]),
        cores=[Core(accelerators=config[core]) for core in cores],
    )
