from collections.abc import Sequence

from xdsl.ir import (
    Dialect,
    NoTerminator,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    region_def,
    traits_def,
    var_operand_def,
)


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
class StageOp(IRDLOperation):
    """
    A single pipeline stage. The operation tracks
    which buffers (= pipeline registers) the operations
    in its region works on.
    """

    name = "pipeline.stage"

    input_buffers = var_operand_def()
    output_buffers = var_operand_def()

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator())

    def __init__(
        self,
        input_buffers: Sequence[SSAValue | Operation],
        output_buffers: Sequence[SSAValue | Operation],
        body: Region,
    ) -> None:
        # input and output buffers should be alloc operations
        super().__init__(operands=[input_buffers, output_buffers], regions=[body])


@irdl_op_definition
class DoubleStageOp(IRDLOperation):
    """
    A single pipeline stage, working on two sets of
    buffers (= pipeline registers). In even cycles, the
    even buffers will be used. In odd cycles, the odd buffers
    will be used. This allows for an unrolling of the pipeline.

    (cycle does not correspond to clock cycle, but pipeline cycle)
    """

    name = "pipeline.double_stage"

    input_buffers_even = var_operand_def()
    input_buffers_odd = var_operand_def()

    output_buffers_even = var_operand_def()
    output_buffers_odd = var_operand_def()

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator())

    def __init__(
        self,
        input_buffers_even: Sequence[SSAValue | Operation],
        input_buffers_odd: Sequence[SSAValue | Operation],
        output_buffers_even: Sequence[SSAValue | Operation],
        output_buffers_odd: Sequence[SSAValue | Operation],
        body: Region,
    ) -> None:
        # input and output buffers should be alloc operations
        super().__init__(
            operands=[
                input_buffers_even,
                input_buffers_odd,
                output_buffers_even,
                output_buffers_odd,
            ],
            regions=[body],
        )


Pipeline = Dialect(
    "pipeline",
    [
        # ops
        PipelineOp,
        StageOp,
        DoubleStageOp,
    ],
    [
        # attribut
    ],
)
