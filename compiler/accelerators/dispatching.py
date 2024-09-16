from abc import ABC
from collections.abc import Iterable

from xdsl.ir import Attribute

from compiler.accelerators.accelerator import Accelerator
from compiler.dialects.kernel import KernelOp


class SupportedKernel:
    type: type[KernelOp]
    operand_types: list[Attribute]

class DispatchTemplate(Accelerator, ABC):
    """
    Specifies a dispatching template to dispatch linalg generic kernels to accelerators.
    """

    supported_kernels: Iterable[SupportedKernel]


