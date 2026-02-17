from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr, IntegerType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def, result_def
from xdsl.parser import Parser
from xdsl.printer import Printer


class HardfloatOperation(IRDLOperation, ABC):
    res = result_def(IntegerType)
    sig_width = prop_def(IntAttr)
    exp_width = prop_def(IntAttr)

    def __init__(
        self,
        operands: Sequence[Operation | SSAValue],
        result_types: Sequence[Attribute],
        sig_width: int,
        exp_width: int,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=operands,
            result_types=result_types,
            attributes=attr_dict,
            properties={"exp_width": IntAttr(sig_width), "sig_width": IntAttr(exp_width)},
        )

    def print(self, printer: Printer):
        printer.print_string(f"<{self.sig_width.data}, {self.exp_width.data}>")
        printer.print_operands(self.operands)
        printer.print_string(" : ")
        printer.print_function_type(input_types=self.operand_types, output_types=self.result_types)

    @classmethod
    def parse(cls, parser: Parser) -> HardfloatOperation:
        parser.parse_punctuation("<")
        sig_width = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_punctuation(",")
        exp_width = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_punctuation(">")
        operands = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.PAREN, parse=parser.parse_unresolved_operand
        )
        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        in_types = func_type.inputs.data
        out_types = func_type.outputs.data
        res_operands: list[SSAValue[Attribute]] = []
        for opnd, typ in zip(operands, in_types, strict=True):
            res_operands.append(parser.resolve_operand(opnd, typ))
        return cls(operands=res_operands, result_types=out_types, sig_width=sig_width, exp_width=exp_width)


class UnaryHardfloatOp(HardfloatOperation, ABC):
    input = operand_def(IntegerType)


class BinaryHardfloatOp(HardfloatOperation, ABC):
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)


@irdl_op_definition
class MulOp(BinaryHardfloatOp):
    name = "hardfloat.mul"


@irdl_op_definition
class AddOp(BinaryHardfloatOp):
    name = "hardfloat.add"


@irdl_op_definition
class RecodeOp(UnaryHardfloatOp):
    name = "hardfloat.recode"


@irdl_op_definition
class UnrecodeOp(UnaryHardfloatOp):
    name = "hardfloat.unrecode"


Hardfloat = Dialect(
    "hardfloat",
    [
        MulOp,
        AddOp,
        RecodeOp,
        UnrecodeOp,
    ],
)
