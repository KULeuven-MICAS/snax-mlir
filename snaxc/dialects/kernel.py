from abc import ABC
from collections.abc import Sequence
from typing import cast

from xdsl.builder import Builder
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import I8, I32, BoolAttr, DenseArrayBase, IntegerType
from xdsl.ir import Attribute, BlockArgument, Dialect, Operation, Region, SSAValue
from xdsl.irdl import attr_def, irdl_op_definition, operand_def, result_def
from xdsl.parser import IntegerAttr, IRDLOperation


class KernelOp(IRDLOperation, ABC):
    """
    Operation representing different types of kernel operations of a
    linalg generic body that can be executed by accelerators.
    """

    ...


class Parsable(ABC):
    @property
    def equivalent_region(self) -> Region: ...


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
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"

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
            mul = arith.MuliOp(args[0], args[1])
            linalg.YieldOp(mul)

        return equivalent_region


@irdl_op_definition
class AddOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.add"
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"

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
            add = arith.AddiOp(args[0], args[1])
            linalg.YieldOp(add)

        return equivalent_region


@irdl_op_definition
class MacOp(KernelOp, BinaryOp, Parsable):
    name = "kernel.mac"
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"

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
            mul = arith.MuliOp(args[0], args[1])
            mac = arith.AddiOp(args[2], mul)
            linalg.YieldOp(mac)

        @Builder.implicit_region(
            (
                SSAValue.get(self.lhs).type,
                SSAValue.get(self.rhs).type,
                *self.result_types,
            )
        )
        def equivalent_region_extsi(args: tuple[BlockArgument, ...]) -> None:
            a = arith.ExtSIOp(args[0], cast(IntegerType, args[2].type))
            b = arith.ExtSIOp(args[1], cast(IntegerType, args[2].type))
            mul = arith.MuliOp(a, b)
            mac = arith.AddiOp(args[2], mul)
            linalg.YieldOp(mac)

        if SSAValue.get(self.lhs).type == self.result_types[0]:
            return equivalent_region
        else:
            return equivalent_region_extsi


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
            subi_lhs = arith.SubiOp(extsi_lhs, args[2])
            assert isinstance(zp_rhs_type := args[3].type, IntegerType)
            extsi_rhs = arith.ExtSIOp(args[1], zp_rhs_type)
            subi_rhs = arith.SubiOp(extsi_rhs, args[3])
            mul = arith.MuliOp(subi_lhs, subi_rhs)
            mac = arith.AddiOp(args[4], mul)
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
    multiplier = attr_def(DenseArrayBase)
    shift = attr_def(DenseArrayBase)
    max_int = attr_def(IntegerAttr[I8])
    min_int = attr_def(IntegerAttr[I8])
    double_round = attr_def(BoolAttr)

    assembly_format = "$input attr-dict `:` `(` type($input) `)` `->` type($result)"

    def __init__(
        self,
        input: SSAValue | Operation,
        result_type: Attribute,
        input_zp: int | IntegerAttr[I8],
        output_zp: int | IntegerAttr[I8],
        multiplier: Sequence[int] | Sequence[IntegerAttr[I32]] | DenseArrayBase,
        shift: Sequence[int] | Sequence[IntegerAttr[I8]] | DenseArrayBase,
        max_int: int | IntegerAttr[I8],
        min_int: int | IntegerAttr[I8],
        double_round: bool | BoolAttr = False,
    ):
        input = SSAValue.get(input)
        if isinstance(input_zp, int):
            input_zp = IntegerAttr.from_int_and_width(input_zp, 8)
        if isinstance(output_zp, int):
            output_zp = IntegerAttr.from_int_and_width(output_zp, 8)
        if not isinstance(multiplier, DenseArrayBase):
            multiplier = DenseArrayBase.create_dense_int(
                IntegerType(32), [x if isinstance(x, int) else x.value.data for x in multiplier]
            )
        if not isinstance(shift, DenseArrayBase):
            shift = DenseArrayBase.create_dense_int(
                IntegerType(32), [x if isinstance(x, int) else x.value.data for x in shift]
            )
        if isinstance(max_int, int):
            max_int = IntegerAttr.from_int_and_width(max_int, 8)
        if isinstance(min_int, int):
            min_int = IntegerAttr.from_int_and_width(min_int, 8)
        if isinstance(double_round, bool):
            double_round = IntegerAttr.from_int_and_width(1 if double_round else 0, 1)
        super().__init__(
            operands=[input],
            result_types=[result_type],
            attributes={
                "input_zp": input_zp,
                "output_zp": output_zp,
                "multiplier": multiplier,
                "shift": shift,
                "max_int": max_int,
                "min_int": min_int,
                "double_round": double_round,
            },
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
