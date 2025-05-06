from collections.abc import Sequence
from typing import Self

from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.dialects.scf import ForOp
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import HasParent, IsTerminator, NoTerminator, Pure


@irdl_op_definition
class PipelineOp(IRDLOperation):
    """
    An operation whose region is a sequence of one
    index op and or more pipeline stages.
    """

    name = "pipeline.pipeline"

    body = region_def("single_block")

    traits = traits_def(NoTerminator(), HasParent(ForOp))

    assembly_format = "$body attr-dict"

    def __init__(self, body: Region) -> None:
        super().__init__(regions=[body])


@irdl_op_definition
class IndexOp(IRDLOperation):
    """
    An index op is the collection of operations without explicit side-effects
    and not dispatchable to a given accelerator.
    For example, arith ops, memref subviews, ...
    These are often computed using the for loop variable `i`.
    By wrapping them in a pipeline.index op, it is easy to get these values
    for other values than `i`, such as `i+1`, `i+2`, etc.
    """

    name = "pipeline.index"

    input = operand_def(IndexType)
    outputs = var_result_def()

    body = region_def("single_block")

    traits = traits_def(HasParent(PipelineOp))

    assembly_format = "$input `->` type($outputs) $body attr-dict"

    def __init__(
        self,
        input: SSAValue | Operation,
        result_types: Sequence[Attribute],
        body: Region,
    ):
        super().__init__(operands=[input], result_types=[result_types], regions=[body])

    @classmethod
    def empty(cls, input: SSAValue | Operation) -> Self:
        """
        Create an empty index op with the given input.
        """
        return cls(input=input, result_types=[], body=Region(Block([YieldOp()])))


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    A yield operation for the pipeline index op.
    """

    name = "pipeline.yield"

    traits = traits_def(IsTerminator(), HasParent(IndexOp), Pure())


@irdl_op_definition
class StageOp(IRDLOperation):
    """
    A single pipeline stage. The operation tracks which input / output buffers defined
    outside of the pipleine op the operations in its region works on.
    """

    name = "pipeline.stage"

    ins = var_operand_def()
    outs = var_operand_def()

    body = region_def("single_block")

    index = prop_def(IntegerAttr[IndexType])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    assembly_format = (
        "$index (` ins` `(` $ins^ `:` type($ins) `)`)?"
        + "(` outs` `(` $outs^ `:` type($outs) `)`)? $body attr-dict"
    )

    traits = traits_def(NoTerminator(), HasParent(PipelineOp))

    def __init__(
        self,
        ins: Sequence[SSAValue | Operation],
        outs: Sequence[SSAValue | Operation],
        index: int | IntegerAttr[IndexType],
        body: Region,
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_index_int_value(index)
        super().__init__(
            operands=[ins, outs], regions=[body], properties={"index": index}
        )


Pipeline = Dialect(
    "pipeline",
    [
        PipelineOp,
        IndexOp,
        YieldOp,
        StageOp,
    ],
    [],
)
