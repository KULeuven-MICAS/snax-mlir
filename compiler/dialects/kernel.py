from abc import ABC

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect
from xdsl.irdl import operand_def
from xdsl.irdl.operations import irdl_op_definition, result_def
from xdsl.parser import IRDLOperation


class KernelOp(IRDLOperation, ABC):
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
    acc = operand_def(IntegerType)
    assembly_format = (
        "$lhs `,` $rhs `acc``:` $acc attr-dict `:`"
        "type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)"
    )


@irdl_op_definition
class QMacOp(KernelOp, QuantizedBinaryOp):
    name = "kernel.qmac"
    acc = operand_def(IntegerType)
    assembly_format = (
        "$lhs `,` $rhs `acc` `:` $acc `zp_lhs``:` $zp_lhs `zp_rhs``:` $zp_rhs attr-dict `:`"
        " type($lhs) `,` type($rhs) `,` type($acc) `,` type($zp_lhs) `,` type($zp_rhs) `->` type($result)"
    )


Kernel = Dialect("kernel", [MulOp, AddOp, MacOp, QMacOp])
