from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr, IntegerType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def, result_def, traits_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import OpTrait, Pure
from xdsl.utils.exceptions import VerifyException


class HardfloatOperation(IRDLOperation, ABC):
    res = result_def(IntegerType)
    sig_width = prop_def(IntAttr)
    exp_width = prop_def(IntAttr)
    traits = traits_def(Pure())

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
            properties={"exp_width": IntAttr(exp_width), "sig_width": IntAttr(sig_width)},
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


class RecodedInputs(OpTrait):
    """
    Provides verification to HardfloatOperation's to have proper bitwidths for their recoded inputs
    """

    def verify(self, op: Operation) -> None:
        if not isinstance(op, HardfloatOperation):
            raise VerifyException("Expect op to subclass HardfloatOperation")
        sig_width = op.sig_width.data
        exp_width = op.exp_width.data
        for typ in op.operand_types:
            if not isinstance(typ, IntegerType):
                raise VerifyException("Expect input type to be of IntegerType")
            if typ.bitwidth != sig_width + exp_width + 1:
                raise VerifyException(
                    f"Expect input type to be of sig_width ({sig_width}) + exp_width ({exp_width}) + 1 = {typ.bitwidth}"
                )


class RecodedOutputs(OpTrait):
    """
    Provides verification to HardfloatOperation's to have proper bitwidths for their recoded outputs
    """

    def verify(self, op: Operation) -> None:
        if not isinstance(op, HardfloatOperation):
            raise VerifyException("Expect op to subclass HardfloatOperation")
        sig_width = op.sig_width.data
        exp_width = op.exp_width.data
        for typ in op.result_types:
            if not isinstance(typ, IntegerType):
                raise VerifyException("Expect input type to be of IntegerType")
            if typ.bitwidth != sig_width + exp_width + 1:
                raise VerifyException(
                    f"Expect input type to be of sig_width ({sig_width}) + exp_width ({exp_width}) + 1 = {typ.bitwidth}"
                )


class UnaryHardfloatOp(HardfloatOperation, ABC):
    input = operand_def(IntegerType)


class BinaryHardfloatOp(HardfloatOperation, ABC):
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)


@irdl_op_definition
class MulRecFnOp(BinaryHardfloatOp):
    name = "hardfloat.mul_rec_fn"
    traits = traits_def(RecodedInputs(), RecodedOutputs())


@irdl_op_definition
class AddRecFnOp(BinaryHardfloatOp):
    name = "hardfloat.add_rec_fn"
    traits = traits_def(RecodedInputs(), RecodedOutputs())


@irdl_op_definition
class FnToRecFnOp(UnaryHardfloatOp):
    name = "hardfloat.fn_to_rec_fn"
    traits = traits_def(RecodedOutputs())


@irdl_op_definition
class RecFnToFnOp(UnaryHardfloatOp):
    name = "hardfloat.rec_fn_to_fn"
    traits = traits_def(RecodedInputs())


Hardfloat = Dialect(
    "hardfloat",
    [
        MulRecFnOp,
        AddRecFnOp,
        RecFnToFnOp,
        FnToRecFnOp,
    ],
)
