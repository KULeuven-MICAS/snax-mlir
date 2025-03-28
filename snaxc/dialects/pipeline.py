from collections.abc import Sequence

from typing_extensions import Self
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    NoTerminator,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import HasParent, IsTerminator


@irdl_op_definition
class PipelineOp(IRDLOperation):
    """
    An operation whose region is a sequence of one
    or more pipeline stages.
    """

    name = "pipeline.pipeline"

    body = region_def("single_block")

    traits = traits_def(NoTerminator())

    def __init__(self, body: Region) -> None:
        super().__init__(regions=[body])


@irdl_op_definition
class YieldOp(IRDLOperation):
    """
    A yield operation for the pipeline index op.
    """

    name = "pipeline.yield"

    arguments = var_operand_def()

    traits = traits_def(IsTerminator(), HasParent(PipelineOp))

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(operands=[operands])

@irdl_op_definition
class IndexOp(IRDLOperation):
    """
    All of the subviews to larger constants
    in the pipeline.
    """
    name = "pipeline.index"

    input = operand_def()
    test = var_result_def()

    body = region_def()

    def __init__(self, input: SSAValue | Operation, result_types: Sequence[Attribute], body: Region):
        super().__init__(operands=[input], result_types=[result_types], regions=[body])

    @classmethod
    def empty(cls, input: SSAValue | Operation) -> Self:
        input = SSAValue.get(input)
        return cls(input, [], Region(Block([YieldOp()], arg_types=[input.type])))


@irdl_op_definition
class StageOp(IRDLOperation):
    """
    A single pipeline stage. The operation tracks
    which buffers (= pipeline registers) the operations
    in its region works on.
    """

    name = "pipeline.stage"

    buffers = var_operand_def()

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator())

    def __init__(
        self,
        buffers: Sequence[SSAValue | Operation],
        body: Region,
    ) -> None:
        # input and output buffers should be alloc operations
        super().__init__(operands=[buffers], regions=[body])

Pipeline = Dialect(
    "pipeline",
    [
        PipelineOp,
        StageOp,
        IndexOp,
        YieldOp,
    ],
    [],
)
