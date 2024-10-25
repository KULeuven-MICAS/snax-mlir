from collections.abc import Sequence
from math import prod

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap

from compiler.accelerators.dispatching import DispatchTemplate, SupportedKernel
from compiler.accelerators.snax import (
    SNAXAccelerator,
)
from compiler.dialects import accfg, kernel


class SNAXXDMAAccelerator(SNAXAccelerator, DispatchTemplate):
    """
    Accelerator Interface class for SNAX XDMA accelerator
    """

    name = "snax_xdma"

    supported_kernels = (SupportedKernel(kernel.MaxOp, (i8, i8, i8)),)

    def __init__(self) -> None:
        self.fields = tuple()

        self.launch_fields = tuple()

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX XDMA accelerator op
        """

        op = accfg.AcceleratorOp(
            self.name,
            {},
            {},
            0,
        )
        return op

    # do not convert to acc ops
    def convert_to_acc_ops(self, op: None) -> Sequence[Operation]:
        return []


    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        return []
 
