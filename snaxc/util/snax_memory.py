from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr


@dataclass(frozen=True)
class SnaxMemory:
    # MLIR memory space attribute
    attribute: StringAttr

    # Memory capacity in bytes
    capacity: int

    # Memory starting address
    start: int


# define the available memory spaces in the SNAX Cluster
L1 = SnaxMemory(StringAttr("L1"), capacity=65536, start=0x10000000)
L3 = SnaxMemory(StringAttr("L3"), capacity=int(1e9), start=0x80000000)
