from abc import ABC

from xdsl.builder import Builder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import I8, I32, BoolAttr, IntegerType
from xdsl.ir import BlockArgument, Dialect, Region, SSAValue
from xdsl.irdl import attr_def, operand_def
from xdsl.irdl.operations import irdl_op_definition, result_def
from xdsl.parser import IntegerAttr, IRDLOperation


class KernelOp(IRDLOperation, ABC):
    """
    Operation representing different types of kernel operations of a
    linalg generic body that can be executed by accelerators.
    """

    ...


class Parsable(ABC):
    @property
    def equivalent_region(self) -> Region:
        ...


class BinaryOp:
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    result = result_def(IntegerType)


class QuantizedBinaryOp:
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    zp_lhs = operand_def(IntegerType)
    zp_rhs = operand_def(IntegerType)
    result = result_def(IntegerType)


@irdl_op_definition
class MulOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.mul"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def equivalent_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def equivalent_region(args: tuple[BlockArgument, ...]) -> None:
            mul = arith.Muli(args[0], args[1])
            linalg.YieldOp(mul)

        return equivalent_region


@irdl_op_definition
class AddOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.add"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def equivalent_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def equivalent_region(args: tuple[BlockArgument, ...]) -> None:
            add = arith.Addi(args[0], args[1])
            linalg.YieldOp(add)

        return equivalent_region


@irdl_op_definition
class MacOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.mac"
    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    @property
    def equivalent_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def equivalent_region(args: tuple[BlockArgument, ...]) -> None:
            mul = arith.Muli(args[0], args[1])
            mac = arith.Addi(args[2], mul)
            linalg.YieldOp(mac)

        return equivalent_region


@irdl_op_definition
class QMacOp(KernelOp, QuantizedBinaryOp, Parsable):
    name = "kernel.qmac"
    assembly_format = (
        "$lhs `,` $rhs `zp_lhs``:` $zp_lhs `zp_rhs``:` $zp_rhs attr-dict `:`"
        " type($lhs) `,` type($rhs) `,` type($zp_lhs) `,` type($zp_rhs) `->` type($result)"
    )

    @property
    def equivalent_region(self) -> Region:
        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                SSAValue.get(self.zp_lhs).type,
                SSAValue.get(self.zp_rhs).type,
                *self.result_types,
            )
        )
        def equivalent_region(args: tuple[BlockArgument, ...]) -> None:
            assert isinstance(zp_lhs_type := args[2].type, IntegerType)
            extsi_lhs = arith.ExtSIOp(args[0], zp_lhs_type)
            subi_lhs = arith.Subi(extsi_lhs, args[2])
            assert isinstance(zp_rhs_type := args[3].type, IntegerType)
            extsi_rhs = arith.ExtSIOp(args[1], zp_rhs_type)
            subi_rhs = arith.Subi(extsi_rhs, args[3])
            mul = arith.Muli(subi_lhs, subi_rhs)
            mac = arith.Addi(args[4], mul)
            linalg.YieldOp(mac)

        return equivalent_region


@irdl_op_definition
class RescaleOp(KernelOp):
    """
    Operation applying rescaling according to the spec in
    https://gist.github.com/jorendumoulin/83352a1e84501ec4a7b3790461fee2bf
    """

    name = "kernel.rescale"

    input = operand_def(IntegerType)
    result = result_def(IntegerType)

    input_zp = attr_def(IntegerAttr[I8])
    output_zp = attr_def(IntegerAttr[I8])
    multiplier = attr_def(IntegerAttr[I32])
    shift = attr_def(IntegerAttr[I8])
    max_int = attr_def(IntegerAttr[I8])
    min_int = attr_def(IntegerAttr[I8])
    double_round = attr_def(BoolAttr)

    assembly_format = (
        "$input attr-dict `zero_points` `(` $input_zp `,` $output_zp `)`"
        "`rescale` `(` $multiplier ` ` `>` `>` $shift `)` `clamp` `(` $min_int `,` $max_int `)`"
        " `double_round` `=` $double_round `:` type($input) `->` type($result)"
    )


Kernel = Dialect(
    "kernel",
    [
        MulOp,
        AddOp,
        MacOp,
        QMacOp,
        RescaleOp,
    ],
)
