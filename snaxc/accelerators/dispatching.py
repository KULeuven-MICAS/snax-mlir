from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.ir import Attribute, Operation

from snaxc.accelerators.accelerator import Accelerator
from snaxc.dialects.kernel import KernelOp


@dataclass
class SupportedKernel:
    kernel_type: type[KernelOp]
    operand_types: Iterable[Attribute]

    def is_same_kernel(self, kernel_op: Operation | None) -> bool:
        """
        Check if the kernel operation matches the supported kernel type and operand types.
        """
        if kernel_op is None:
            return False
        if not isinstance(kernel_op, KernelOp):
            return False
        return self._check_kernel_type(kernel_op) and self._check_attribute_types(
            kernel_op
        )

    def _check_kernel_type(self, kernel_op: KernelOp) -> bool:
        """
        Check if the kernel operation is of the supported type.
        """
        return isinstance(kernel_op, self.kernel_type)

    def _check_attribute_types(self, kernel_op: KernelOp) -> bool:
        """
        Check if the attributes of the kernel operation match the expected types.
        """
        expected_types = list(self.operand_types)
        i = 0
        for operand_type in kernel_op.operand_types:
            if operand_type != expected_types[i]:
                return False
            i += 1
        for result_type in kernel_op.result_types:
            if result_type != expected_types[i]:
                return False
            i += 1
        if i != len(expected_types):
            return False
        return True


@dataclass
class DispatchTemplate(Accelerator, ABC):
    """
    Specifies a dispatching template to dispatch linalg generic kernels to accelerators.
    """

    supported_kernels: Iterable[SupportedKernel]
