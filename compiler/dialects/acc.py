from collections.abc import Iterable, Sequence

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

    name = "acc.token"


@irdl_attr_definition
class StateType(ParametrizedAttribute, TypeAttribute):
    """
    Used to trace an accelerators CSR state through def-use chain
    """

    name = "acc.state"

    accelerator: ParameterDef[StringAttr]


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """
    Launch an accelerator. This acts as a barrier for CSR values,
    meaning CSRs can be safely modified after a launch op without
    interfering with the Accelerator.
    """

    name = "acc.launch"

    accelerator = prop_def(StringAttr)

    token = result_def()

    def verify_(self) -> None:
        # that the token is used
        if len(self.token.uses) != 1 or not isinstance(
            next(iter(self.token.uses)).operation, AwaitOp
        ):
            raise VerifyException("Launch token must be used by exactly one await op")
        # TODO: allow use in control flow


@irdl_op_definition
class AwaitOp(IRDLOperation):
    name = "acc.await"

    token = operand_def(TokenType)


@irdl_op_definition
class SetupOp(IRDLOperation):
    name = "acc.setup"

    values = var_operand_def(Attribute)  # TODO: make more precise?
    """
    The actual values used to set up the CSRs
    """

    in_state = opt_operand_def(StateType)
    """
    The state produced by a previous acc.setup
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
        vals: Sequence[SSAValue],
        param_names: Sequence[str],
        accelerator: str | StringAttr,
        in_state: SSAValue | Operation | None = None,
    ):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)

        super().__init__(
            operands=[vals, in_state],
            properties={
                "param_names": ArrayAttr([StringAttr(x) for x in param_names]),
                "accelerator": accelerator,
            },
            result_types=[StateType((accelerator,))],
        )

    def iter_params(self) -> Iterable[tuple[str, SSAValue]]:
        return zip((p.data for p in self.param_names), self.values)

    def verify_(self) -> None:
        # that accelerator on input matches output
        if self.in_state is not None:
            if self.in_state.type != self.out_state.type:
                raise VerifyException("Input and output state accelerators must match")
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

        # that a state is not used twice
        if len(self.out_state.uses) > 1:
            raise VerifyException("States must be used at most once")


ACC = Dialect(
    "acc",
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
