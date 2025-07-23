from abc import ABC
from typing import Self

from snaxc.accelerators.accelerator import Accelerator
from snaxc.tools.configs import AcceleratorConfig


class ConfigurableAccelerator(Accelerator, ABC):
    @classmethod
    def from_config(cls, config: AcceleratorConfig) -> Self:
        return cls()
