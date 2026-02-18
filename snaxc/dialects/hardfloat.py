from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import IntAttr, IntegerType, Signedness, SignednessAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def, traits_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import OpTrait, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


class HardfloatOperation(IRDLOperation, ABC):
    res = result_def(IntegerType)
    sig_width = prop_def(IntAttr)
    exp_width = prop_def(IntAttr)
    int_width = opt_prop_def(IntAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        operands: Sequence[Operation | SSAValue],
        result_types: Sequence[Attribute],
        sig_width: int,
        exp_width: int,
        int_width: int | None = None,
        attr_dict: dict[str, Attribute] | None = None,
        prop_dict: dict[str, Attribute] | None = None,
    ):
        props: dict[str, Attribute] = {}
        props.update({"exp_width": IntAttr(exp_width), "sig_width": IntAttr(sig_width)})
        if int_width is not None:
            props["int_width"] = IntAttr(int_width)
        if prop_dict is not None:
            props.update(prop_dict)
        super().__init__(operands=operands, result_types=result_types, attributes=attr_dict, properties=props)

    def print(self, printer: Printer):
        if self.int_width is not None:
            printer.print_string(f"<{self.sig_width.data}, {self.exp_width.data}, {self.int_width.data}>")
        else:
            printer.print_string(f"<{self.sig_width.data}, {self.exp_width.data}>")
        printer.print_operands(self.operands)
        opt_properties = {
            name: attr for name, attr in self.properties.items() if name not in ["sig_width", "exp_width", "int_width"]
        }
        if len(opt_properties) > 0:
            printer.print_string(" ")
            with printer.in_angle_brackets():
                printer.print_attr_dict(opt_properties)
        printer.print_string(" : ")
        printer.print_function_type(input_types=self.operand_types, output_types=self.result_types)

    @classmethod
    def parse(cls, parser: Parser) -> HardfloatOperation:
        parser.parse_punctuation("<")
        sig_width = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_punctuation(",")
        exp_width = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_optional_punctuation(",")
        int_width = parser.parse_optional_integer(allow_boolean=False, allow_negative=False)
        parser.parse_punctuation(">")
        operands = parser.parse_comma_separated_list(
            delimiter=parser.Delimiter.PAREN, parse=parser.parse_unresolved_operand
        )
        prop_dict = parser.parse_optional_properties_dict()
        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        in_types = func_type.inputs.data
        out_types = func_type.outputs.data
        res_operands: list[SSAValue[Attribute]] = []
        for opnd, typ in zip(operands, in_types, strict=True):
            res_operands.append(parser.resolve_operand(opnd, typ))
        return cls(
            operands=res_operands,
            result_types=out_types,
            sig_width=sig_width,
            exp_width=exp_width,
            int_width=int_width,
            prop_dict=prop_dict,
        )


class HardfloatOpTrait(OpTrait, ABC):
    """Base trait for all Hardfloat-specific verification logic."""

    def verify(self, op: Operation) -> None:
        if not isinstance(op, HardfloatOperation):
            raise VerifyException(f"Trait {self.__class__.__name__} expects a HardfloatOperation.")
        self.verify_hardfloat(op)

    @abstractmethod
    def verify_hardfloat(self, op: HardfloatOperation) -> None:
        """Override this instead of verify()"""
        pass


class RecodedInputs(HardfloatOpTrait):
    """
    Provides verification to HardfloatOperation's to have proper bitwidths for their recoded inputs
    """

    def verify_hardfloat(self, op: HardfloatOperation) -> None:
        sig_width = op.sig_width.data
        exp_width = op.exp_width.data
        for typ in op.operand_types:
            if not isinstance(typ, IntegerType):
                raise VerifyException("Expect input type to be of IntegerType")
            if typ.bitwidth != sig_width + exp_width + 1:
                raise VerifyException(
                    f"Expect input type to be of sig_width ({sig_width}) + exp_width ({exp_width}) + 1 = {typ.bitwidth}"
                )


class RecodedOutputs(HardfloatOpTrait):
    """
    Provides verification to HardfloatOperation's to have proper bitwidths for their recoded outputs
    """

    def verify_hardfloat(self, op: HardfloatOperation) -> None:
        sig_width = op.sig_width.data
        exp_width = op.exp_width.data
        for typ in op.result_types:
            typ = cast(IntegerType, typ)  # verified by IRDL
            if typ.bitwidth != sig_width + exp_width + 1:
                raise VerifyException(
                    f"Expect output type to be of sig_width ({sig_width})"
                    f" + exp_width ({exp_width}) + 1 = {typ.bitwidth}"
                )


class IntegerOutputs(HardfloatOpTrait):
    def verify_hardfloat(self, op: HardfloatOperation) -> None:
        if op.int_width is None:
            raise VerifyException("Expect op to have int_width property")
        for typ in op.result_types:
            typ = cast(IntegerType, typ)  # verified by IRDL
            if typ.bitwidth != op.int_width.data:
                raise VerifyException(
                    f"Expect output type ({typ}) to have bitwidth given by int_width property ({op.int_width.data})"
                )


class IntegerInputs(HardfloatOpTrait):
    def verify_hardfloat(self, op: HardfloatOperation) -> None:
        if op.int_width is None:
            raise VerifyException("Expect op to have int_width property")
        for typ in op.operand_types:
            typ = cast(IntegerType, typ)  # verified by IRDL
            if typ.bitwidth != op.int_width.data:
                raise VerifyException(
                    f"Expect input type ({typ}) to have bitwidth given by int_width property ({op.int_width.data})"
                )


class IntegerConversion(OpTrait):
    def verify(self, op: Operation):
        if "signedness" not in op.properties:
            raise VerifyException("IntegerConversion optrait expects signedness attr")
        signedness = op.properties["signedness"]
        if not isa(signedness, SignednessAttr):
            raise VerifyException("Expect property signedness to have type SignednessAttr")
        if signedness.data == Signedness.SIGNLESS:
            raise VerifyException("Property signedness can not be Signless")


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


@irdl_op_definition
class InToRecFnOp(UnaryHardfloatOp):
    name = "hardfloat.in_to_rec_fn"
    signedness = prop_def(SignednessAttr)
    traits = traits_def(IntegerInputs(), RecodedOutputs(), IntegerConversion())


@irdl_op_definition
class RecFnToInOp(UnaryHardfloatOp):
    name = "hardfloat.rec_fn_to_in"
    signedness = prop_def(SignednessAttr)
    traits = traits_def(RecodedInputs(), IntegerOutputs(), IntegerConversion())


Hardfloat = Dialect(
    "hardfloat",
    [
        MulRecFnOp,
        AddRecFnOp,
        RecFnToFnOp,
        FnToRecFnOp,
        InToRecFnOp,
        RecFnToInOp,
    ],
)
