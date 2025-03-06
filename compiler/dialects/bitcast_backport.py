# Backport of upstream arith.bitcast operation
from xdsl.dialects.arith import Arith, signlessIntegerLike, floatingPointLike
from xdsl.dialects.builtin import MemRefType, IndexType, ContainerType
from xdsl.irdl import (
        irdl_op_definition,
        IRDLOperation,
        operand_def,
        result_def,
    )
from xdsl.ir import Operation, Attribute, SSAValue


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "arith.bitcast"

    input = operand_def(
        signlessIntegerLike
        | floatingPointLike
        | MemRefType
    )
    result = result_def(
        signlessIntegerLike
        | floatingPointLike
        | MemRefType
    )

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    def __init__(self, in_arg: SSAValue | Operation, target_type: Attribute):
        super().__init__(operands=[in_arg], result_types=[target_type])

dialect = Arith
dialect._operations.append(BitcastOp)
ArithPatched = dialect
