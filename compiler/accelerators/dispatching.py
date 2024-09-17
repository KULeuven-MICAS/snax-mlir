from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.ir import Attribute

from compiler.accelerators.accelerator import Accelerator
from compiler.dialects.kernel import KernelOp


@dataclass
class SupportedKernel:
    kernel_type: type[KernelOp]
    operand_types: Iterable[Attribute]


@dataclass
class DispatchTemplate(Accelerator, ABC):
    """
    Specifies a dispatching template to dispatch linalg generic kernels to accelerators.
    """

    supported_kernels: Iterable[SupportedKernel]
