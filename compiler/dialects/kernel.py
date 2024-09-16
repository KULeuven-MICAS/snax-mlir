from abc import ABC

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect
from xdsl.irdl import operand_def
from xdsl.irdl.operations import irdl_op_definition, result_def
from xdsl.parser import IRDLOperation


class KernelOp(IRDLOperation, ABC):
    """
    Operation representing different types of kernel operations of a
    linalg generic body that can be executed by accelerators.
    """

    ...


class BinaryOp:
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    result = result_def(IntegerType)


class QuantizedBinaryOp(BinaryOp):
    zp_lhs = operand_def(IntegerType)
    zp_rhs = operand_def(IntegerType)


@irdl_op_definition
class MulOp(KernelOp, BinaryOp):
    name = "kernel.mul"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

@irdl_op_definition
class AddOp(KernelOp, BinaryOp):
    name = "kernel.add"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )


@irdl_op_definition
class MacOp(KernelOp, BinaryOp):
    name = "kernel.mac"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )


@irdl_op_definition
class QMacOp(KernelOp, QuantizedBinaryOp):
    name = "kernel.qmac"
    assembly_format = (
        "$lhs `,` $rhs `zp_lhs``:` $zp_lhs `zp_rhs``:` $zp_rhs attr-dict `:`"
        " type($lhs) `,` type($rhs) `,` type($zp_lhs) `,` type($zp_rhs) `->` type($result)"
    )


Kernel = Dialect("kernel", [MulOp, AddOp, MacOp, QMacOp])
