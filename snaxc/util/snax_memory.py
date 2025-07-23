from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr

from snaxc.tools.configs import SnaxMemoryConfig


@dataclass(frozen=True)
class SnaxMemory:
    # MLIR memory space attribute
    attribute: StringAttr

    # Memory capacity in bytes
    capacity: int

    # Memory starting address
    start: int

    @classmethod
    def from_config(cls, config: SnaxMemoryConfig) -> "SnaxMemory":
        """
        Create a SnaxMemory instance from a SnaxMemoryConfig.
        """
        return cls(
            attribute=StringAttr(config.name),
            capacity=config.size,
            start=config.start,
        )


# define the available memory spaces in the SNAX Cluster
L1 = SnaxMemory(StringAttr("L1"), capacity=65536, start=0x10000000)
L3 = SnaxMemory(StringAttr("L3"), capacity=int(1e9), start=0x80000000)
TEST = SnaxMemory(StringAttr("Test"), capacity=100, start=0)

_memory_registry: dict[StringAttr, SnaxMemory] = {
    L1.attribute: L1,
    L3.attribute: L3,
    TEST.attribute: TEST,
}
