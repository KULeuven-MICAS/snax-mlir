from __future__ import annotations

from collections.abc import Iterable

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    VerifyException,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import SymbolOpInterface


@irdl_attr_definition
class TokenType(ParametrizedAttribute, TypeAttribute):
    """
    Async token type for launched accelerator requests.
    """

    name = "acc2.token"

    accelerator: ParameterDef[StringAttr]

    def __init__(self, accelerator: str | StringAttr):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)
        return super().__init__([accelerator])


@irdl_attr_definition
class StateType(ParametrizedAttribute, TypeAttribute):
    """
    Used to trace an accelerators CSR state through def-use chain
    """

    name = "acc2.state"

    accelerator: ParameterDef[StringAttr]

    def __init__(self, accelerator: str | StringAttr):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)
        return super().__init__([accelerator])


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """
    Launch an accelerator. This acts as a barrier for CSR values,
    meaning CSRs can be safely modified after a launch op without
    interfering with the Accelerator.
    """

    name = "acc2.launch"

    values = var_operand_def(Attribute)  # TODO: make more precise?
    """
    The actual values used to set up registers linked to launch
    """

    state = operand_def(StateType)

    param_names = prop_def(ArrayAttr[StringAttr])
    """
    Maps the SSA values in `values` to accelerator launch parameters
    """

    accelerator = prop_def(StringAttr)

    token = result_def()

    def __init__(
        self,
        vals: list[SSAValue | Operation],
        param_names: Iterable[str] | Iterable[StringAttr],
        state: SSAValue | Operation,
    ):
        state_val: SSAValue = SSAValue.get(state)

        if not isinstance(state_val.type, StateType):
            raise ValueError("`state` SSA Value must be of type `acc2.state`!")

        param_names_tuple: tuple[StringAttr, ...] = tuple(
            StringAttr(name) if isinstance(name, str) else name for name in param_names
        )
        super().__init__(
            operands=[vals, state],
            properties={
                "param_names": ArrayAttr(param_names_tuple),
                "accelerator": state_val.type.accelerator,
            },
            result_types=[TokenType(state_val.type.accelerator)],
        )

    def iter_params(self) -> Iterable[tuple[str, SSAValue]]:
        return zip((p.data for p in self.param_names), self.values)

    def verify_(self) -> None:
        # that the state and my accelerator match
        assert isinstance(self.state.type, StateType)
        if self.state.type.accelerator != self.accelerator:
            raise VerifyException(
                "The state's accelerator does not match the launch accelerator!"
            )
        # that the token and my accelerator match
        assert isinstance(self.token.type, TokenType)
        if self.token.type.accelerator != self.accelerator:
            raise VerifyException(
                "The token's accelerator does not match the launch accelerator!"
            )

        # that the token is used
        if len(self.token.uses) != 1 or not isinstance(
            next(iter(self.token.uses)).operation, AwaitOp
        ):
            raise VerifyException("Launch token must be used by exactly one await op")

        # that len(values) == len(param_names)
        if len(self.values) != len(self.param_names):
            raise ValueError(
                "Must have received same number of values as parameter names"
            )
        # TODO: allow use in control flow


@irdl_op_definition
class AwaitOp(IRDLOperation):
    """
    Blocks until the launched operation finishes.
    """

    name = "acc2.await"

    token = operand_def(TokenType)

    def __init__(self, token: SSAValue | Operation):
        super().__init__(operands=[token])


@irdl_op_definition
class SetupOp(IRDLOperation):
    """
    acc2.setup writes values to a specific accelerators configuration and returns
    a value representing the currently known state of that accelerator's config.

    If acc2.setup is called without any parameters, the resulting state is the
    "empty" state, that represents a state without known values.
    """

    name = "acc2.setup"

    values = var_operand_def(Attribute)  # TODO: make more precise?
    """
    The actual values used to set up the CSRs
    """

    in_state = opt_operand_def(StateType)
    """
    The state produced by a previous acc2.setup
    """

    out_state = result_def(StateType)
    """
    The CSR state after the setup op modified it.
    """

    param_names = prop_def(ArrayAttr[StringAttr])
    """
    Maps the SSA values in `values` to accelerator parameter names
    """

    accelerator = prop_def(StringAttr)
    """
    Name of the accelerator this setup is for
    """

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        vals: Iterable[SSAValue | Operation],
        param_names: Iterable[str] | Iterable[StringAttr],
        accelerator: str | StringAttr,
        in_state: SSAValue | Operation | None = None,
    ):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)

        param_names_tuple: tuple[StringAttr, ...] = tuple(
            StringAttr(name) if isinstance(name, str) else name for name in param_names
        )

        super().__init__(
            operands=[vals, in_state],
            properties={
                "param_names": ArrayAttr(param_names_tuple),
                "accelerator": accelerator,
            },
            result_types=[StateType(accelerator)],
        )

    def iter_params(self) -> Iterable[tuple[str, SSAValue]]:
        return zip((p.data for p in self.param_names), self.values)

    def verify_(self) -> None:
        # that accelerator on input matches output
        if self.in_state is not None:
            if self.in_state.type != self.out_state.type:
                raise VerifyException("Input and output state accelerators must match")
        assert isinstance(self.out_state.type, StateType)
        if self.accelerator != self.out_state.type.accelerator:
            raise VerifyException(
                "Output state accelerator and accelerator the "
                "operations property must match"
            )

        # that len(values) == len(param_names)
        if len(self.values) != len(self.param_names):
            raise ValueError(
                "Must have received same number of values as parameter names"
            )

    def print(self, printer: Printer):
        printer.print_string(" on ")
        printer.print_string_literal(self.accelerator.data)
        printer.print_string(" (")
        for i, (name, val) in enumerate(zip(self.param_names, self.values)):
            printer.print_string_literal(name.data)
            printer.print_string(" = ")
            printer.print_ssa_value(val)
            printer.print_string(" : ")
            printer.print_attribute(val.type)
            # for all but the last value print separator
            if i != len(self.values) - 1:
                printer.print_string(", ")
        printer.print_string(") ")

        if self.in_state:
            printer.print_string("in_state(")
            printer.print_ssa_value(self.in_state)
            printer.print_string(") ")

        if self.attributes:
            printer.print("attrs ")
            printer.print_attr_dict(self.attributes)
            printer.print(" ")

        printer.print_string(": ")
        printer.print_attribute(self.out_state.type)

    @classmethod
    def parse(cls: type[SetupOp], parser: Parser) -> SetupOp:
        parser.parse_keyword("on")
        accelerator = parser.parse_str_literal("accelerator name")

        def parse_itm() -> tuple[str, SSAValue]:
            name = parser.parse_str_literal("accelerator field name")
            parser.parse_punctuation("=")
            val = parser.parse_operand(f'expected value for field "{name}"')
            parser.parse_punctuation(":")
            typ = parser.parse_type()
            assert (
                val.type == typ
            ), f"ssa value type mismatch! Expected {typ}, got {val.type}"
            return name, val

        args: list[tuple[str, SSAValue]] = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parse_itm
        )

        in_state: SSAValue | None = None
        if parser.parse_optional_keyword("in_state"):
            parser.parse_punctuation("(")
            in_state = parser.parse_operand()
            parser.parse_punctuation(")")

        attributes = {}
        if parser.parse_optional_keyword("attrs"):
            attributes = parser.parse_optional_attr_dict()

        parser.parse_punctuation(":")
        res_typ = parser.parse_type()
        setup_op = cls(
            [val for _, val in args],
            [name for name, _ in args],
            accelerator,
            in_state,
        )
        setup_op.out_state.type = res_typ
        setup_op.attributes.update(attributes)
        return setup_op


class AcceleratorSymbolOpTrait(SymbolOpInterface):
    def get_sym_attr_name(self, op: Operation) -> StringAttr | None:
        assert isinstance(op, AcceleratorOp)
        return StringAttr(op.name_prop.string_value())


@irdl_op_definition
class AcceleratorOp(IRDLOperation):
    """
    Declares an accelerator that can be configures, launched, etc.
    `fields` is a dictionary mapping accelerator configuration names to
    CSR addresses.
    """

    name = "acc2.accelerator"

    traits = frozenset([AcceleratorSymbolOpTrait()])

    name_prop = prop_def(SymbolRefAttr, prop_name="name")

    fields = prop_def(DictionaryAttr)

    launch_fields = prop_def(DictionaryAttr)

    barrier = prop_def(IntegerAttr)  # TODO: this will be reworked in a later version

    def __init__(
        self,
        name: str | StringAttr | SymbolRefAttr,
        fields: dict[str, int] | DictionaryAttr,
        launch_fields: dict[str, int] | DictionaryAttr,
        barrier: int | IntegerAttr,
    ):
        if not isinstance(fields, DictionaryAttr):
            fields = DictionaryAttr(
                {name: IntegerAttr(val, i32) for name, val in fields.items()}
            )

        if not isinstance(launch_fields, DictionaryAttr):
            launch_fields = DictionaryAttr(
                {name: IntegerAttr(val, i32) for name, val in launch_fields.items()}
            )

        super().__init__(
            properties={
                "name": (
                    SymbolRefAttr(name) if not isinstance(name, SymbolRefAttr) else name
                ),
                "fields": fields,
                "launch_fields": launch_fields,
                "barrier": (
                    IntegerAttr(barrier, i32)
                    if not isinstance(barrier, IntegerAttr)
                    else barrier
                ),
            }
        )

    def verify_(self) -> None:
        for _, val in self.fields.data.items():
            if not isinstance(val, IntegerAttr):
                raise VerifyException("fields must only contain IntegerAttr!")

    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields.data.keys())

    def field_items(self) -> Iterable[tuple[str, IntegerAttr]]:
        for name, val in self.fields.data.items():
            assert isinstance(val, IntegerAttr)
            yield name, val

    def launch_field_names(self) -> tuple[str, ...]:
        return tuple(self.launch_fields.data.keys())

    def launch_field_items(self) -> Iterable[tuple[str, IntegerAttr]]:
        for name, val in self.launch_fields.data.items():
            assert isinstance(val, IntegerAttr)
            yield name, val


ACC = Dialect(
    "acc2",
    [
        SetupOp,
        LaunchOp,
        AwaitOp,
        AcceleratorOp,
    ],
    [
        StateType,
        TokenType,
    ],
)
