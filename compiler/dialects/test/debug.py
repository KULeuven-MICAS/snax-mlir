from __future__ import annotations

from xdsl.dialects.builtin import (
    MemRefType,
    StringAttr,
)
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
)


@irdl_op_definition
class DebugOp(IRDLOperation):
    """
    Run a debug statement, passing operands to a C function
    to allow inspection of the memref operands.
    """

    name = "debug.debug"

    op_a = operand_def(MemRefType[Attribute])
    op_b = operand_def(MemRefType[Attribute])
    op_c = operand_def(MemRefType[Attribute])

    # what linalg kernel am i debugging?
    debug_type = prop_def(StringAttr)

    # am i debuggin before or after the op?
    when = prop_def(StringAttr)

    # at what memory level is the data i am inspecting?
    level = prop_def(StringAttr)

    def __init__(
        self,
        op_a: SSAValue,
        op_b: SSAValue,
        op_c: SSAValue,
        debug_type: str,
        when: str,
        level: str = "L1",
    ):
        super().__init__(
            operands=[op_a, op_b, op_c],
            result_types=[],
            properties={
                "debug_type": StringAttr(debug_type),
                "when": StringAttr(when),
                "level": StringAttr(level),
            },
        )


Debug = Dialect("debug", [DebugOp])
