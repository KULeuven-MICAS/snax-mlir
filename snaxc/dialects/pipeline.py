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

    def add_op(self, op: Operation, rewriter: PatternRewriter) -> Self:
        """
        Returns a modified operation with the given op added to the body.
        """

        # get yield op
        assert isinstance(yield_op := self.body.block.last_op, YieldOp)

        # if the op uses the iteration variable as an operand,
        # rewrite it to use the block arg
        # if the op uses the result of this index op,
        # rewrite it to use the internal result
        for i, operand in enumerate(op.operands):
            if operand is self.input:
                op.operands[i] = self.body.block.args[0]
            elif isinstance(operand, OpResult) and operand.op is self:
                op.operands[i] = yield_op.arguments[operand.index]

        rewriter.insert_op(op, InsertPoint.before(yield_op))

        result_starting_index = len(self.result_types)
        new_result_types = list(self.result_types) + list(op.result_types)

        new_yield_op = YieldOp(*yield_op.arguments, *op.results)
        rewriter.replace_op(yield_op, new_yield_op)


        # create new op and insert
        new_op = type(self)(self.input, new_result_types, rewriter.move_region_contents_to_new_regions(self.body))

        # replace results
        for i, result in enumerate(self.results):
            result.replace_by(new_op.results[i])

        # make sure to use this operations result
        for i, result in enumerate(op.results):
            for use in [x for x in result.uses]:
                if use.operation is new_yield_op:
                    continue
                use.operation.operands[use.index] = new_op.results[result_starting_index + i]

        return new_op

    def clear_unused_args(self, rewriter: PatternRewriter) -> Self:

        assert isinstance(yield_op := self.body.block.last_op, YieldOp)

        used_indeces = [i for i, result in enumerate(self.results) if result.uses]

        new_op = type(self)(self.input, [self.result_types[i] for i in used_indeces], rewriter.move_region_contents_to_new_regions(self.body))

        new_yield_op = YieldOp(*[yield_op.arguments[i] for i in used_indeces])

        rewriter.replace_op(yield_op, new_yield_op)

        # replace results
        for new_i, old_i in enumerate(used_indeces):
            self.results[old_i].replace_by(new_op.results[new_i])

        return new_op

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
