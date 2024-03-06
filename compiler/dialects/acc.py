from collections.abc import Iterable

from xdsl.dialects.builtin import ArrayAttr, StringAttr
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


@irdl_attr_definition
class TokenType(ParametrizedAttribute, TypeAttribute):
    """
    Async token type for launched accelerator requests.
    """

    name = "acc2.token"


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

    state = operand_def(StateType)

    accelerator = prop_def(StringAttr)

    token = result_def()

    def __init__(self, state: SSAValue | Operation):
        state_val: SSAValue = SSAValue.get(state)
        if not isinstance(state_val.type, StateType):
            raise ValueError("`state` SSA Value must be of type `acc2.state`!")
        super().__init__(
            operands=[state],
            properties={"accelerator": state_val.type.accelerator},
            result_types=[TokenType()],
        )

    def verify_(self) -> None:
        # that the state and my accelerator match
        assert isinstance(self.state.type, StateType)
        if self.state.type.accelerator != self.accelerator:
            raise VerifyException(
                "The state's accelerator does not match the launch accelerator!"
            )

        # that the token is used
        if len(self.token.uses) != 1 or not isinstance(
            next(iter(self.token.uses)).operation, AwaitOp
        ):
            raise VerifyException("Launch token must be used by exactly one await op")
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


ACC = Dialect(
    "acc2",
    [
        SetupOp,
        LaunchOp,
        AwaitOp,
    ],
    [
        StateType,
        TokenType,
    ],
)
