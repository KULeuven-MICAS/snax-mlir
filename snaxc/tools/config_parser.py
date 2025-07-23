from typing import Any

from dacite import from_dict
from dacite.config import Config as DaciteConfig

from snaxc.accelerators.acc_context import AccContext
from snaxc.accelerators.configurable_accelerator import ConfigurableAccelerator
from snaxc.accelerators.snax_alu import SNAXAluAccelerator
from snaxc.accelerators.snax_gemmx import SNAXGEMMXAccelerator
from snaxc.tools.configs import AcceleratorWrapper, GemmxWrapper, SnaxAluWrapper, SystemConfig
from snaxc.util.snax_memory import SnaxMemory

# mapping the config wrappers to actual accelerators:
config_to_type: dict[type[AcceleratorWrapper], type[ConfigurableAccelerator] | None] = {
    GemmxWrapper: SNAXGEMMXAccelerator,
    SnaxAluWrapper: SNAXAluAccelerator,
}


def parse_config(config: Any) -> AccContext:
    system = from_dict(SystemConfig, config, DaciteConfig(strict=True))
    context = AccContext()
    context.register_memory(SnaxMemory.from_config(system.memory))

    assert len(system.clusters) == 1, "Only one cluster is supported for now"
    cluster = system.clusters[0]

    context.register_memory(SnaxMemory.from_config(cluster.memory))

    for core in cluster.cores:
        for accelerator in core.accelerators:
            # FIXME: dynamic registration of accelerators is unnecessary and annoying here:
            # cannot be dynamic as lifetime of core.gemmx is not guaranteed according to pyright
            accelerator_type = config_to_type.get(type(accelerator))
            if accelerator_type is not None:
                accelerator_instance = accelerator_type.from_config(accelerator.accelerator)
                context.register_accelerator(accelerator_type.name, lambda: accelerator_instance)

    return context
