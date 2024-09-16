from abc import ABC

from xdsl.builder import Builder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import BlockArgument, Dialect, Region, SSAValue
from xdsl.irdl import operand_def
from xdsl.irdl.operations import irdl_op_definition, result_def
from xdsl.parser import IRDLOperation


class KernelOp(IRDLOperation, ABC):
    """
    Operation representing different types of kernel operations of a
    linalg generic body that can be executed by accelerators.
    """

    ...


class Parsable(ABC):
    @property
    def parsing_region(self) -> Region:
        ...


class BinaryOp:
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    result = result_def(IntegerType)


class QuantizedBinaryOp(BinaryOp):
    zp_lhs = operand_def(IntegerType)
    zp_rhs = operand_def(IntegerType)


@irdl_op_definition
class MulOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.mul"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def parsing_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def parsing_region(args: tuple[BlockArgument, ...]) -> None:
            mul = arith.Muli(args[0], args[1])
            linalg.YieldOp(mul)

        return parsing_region


@irdl_op_definition
class AddOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.add"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def parsing_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def parsing_region(args: tuple[BlockArgument, ...]) -> None:
            add = arith.Addi(args[0], args[1])
            linalg.YieldOp(add)

        return parsing_region


@irdl_op_definition
class MacOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.mac"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def parsing_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def parsing_region(args: tuple[BlockArgument, ...]) -> None:
            mul = arith.Muli(args[0], args[1])
            mac = arith.Addi(args[2], mul)
            linalg.YieldOp(mac)

        return parsing_region


@irdl_op_definition
class QMacOp(KernelOp, QuantizedBinaryOp, Parsable):
    name = "kernel.qmac"
    assembly_format = (
        "$lhs `,` $rhs `zp_lhs``:` $zp_lhs `zp_rhs``:` $zp_rhs attr-dict `:`"
        " type($lhs) `,` type($rhs) `,` type($zp_lhs) `,` type($zp_rhs) `->` type($result)"
    )

    @property
    def parsing_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                SSAValue.get(self.zp_lhs).type,
                SSAValue.get(self.zp_rhs).type,
                *self.result_types,
            )
        )
        def parsing_region(args: tuple[BlockArgument, ...]) -> None:
            assert isinstance(zp_lhs_type := args[2].type, IntegerType)
            extsi_lhs = arith.ExtSIOp(args[0], zp_lhs_type)
            subi_lhs = arith.Subi(extsi_lhs, args[2])
            assert isinstance(zp_rhs_type := args[3].type, IntegerType)
            extsi_rhs = arith.ExtSIOp(args[1], zp_rhs_type)
            subi_rhs = arith.Subi(extsi_rhs, args[3])
            mul = arith.Muli(subi_lhs, subi_rhs)
            mac = arith.Addi(args[4], mul)
            linalg.YieldOp(mac)

        return parsing_region


Kernel = Dialect("kernel", [MulOp, AddOp, MacOp, QMacOp])
