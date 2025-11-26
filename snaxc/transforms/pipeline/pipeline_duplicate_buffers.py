from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.dialects.memref import AllocOp
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects.pipeline import IndexOp, PipelineOp, StageOp, YieldOp


@dataclass
class PipelineDuplicateBuffers(RewritePattern):
    """
    This pattern will duplicate buffers used in a pipeline to allow for a full parallel unrolling of the pipeline.
    To accomplish this, the pattern makes use of the input and output buffers of pipeline.stage operations.
    These arguments are the "potentially unsafe" buffers, that can cause read/write conflicts.
    This pattern handles the case, potentially duplicating the buffer if it is used in multiple pipeline stages.
    After this, the buffer is returned as the result of the pipeline index op, and can then be removed from the
    pipeline arguments, as it is now considered "safe" to use.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StageOp, rewriter: PatternRewriter):
        # find a buffer that might need duplication
        if len(op.ins + op.outs) == 0:
            return
        else:
            buffer = (op.ins + op.outs)[0]

        # gather relevant pipeline and index op
        assert isinstance(pipeline_op := op.parent_op(), PipelineOp)
        assert isinstance(index_op := pipeline_op.body.block.first_op, IndexOp)

        # buffers defined by index op can be removed from the stage args
        if isinstance(buffer, OpResult) and buffer.op is index_op:
            # remove from the block
            op.body.block.args[0].replace_by(buffer)
            op.body.block.erase_arg(op.body.block.args[0])
            # remove from the args
            new_stage = StageOp(
                ins=[x for x in op.ins if x is not buffer],
                outs=[x for x in op.outs if x is not buffer],
                index=op.index,
                body=rewriter.move_region_contents_to_new_regions(op.body),
            )
            rewriter.replace_op(op, new_stage)
            return

        in_uses = [
            use
            for use in buffer.uses
            if isinstance(use.operation, StageOp)
            and use.operation.parent_op() is op.parent_op()
            and buffer in use.operation.ins
        ]
        out_uses = [
            use
            for use in buffer.uses
            if isinstance(use.operation, StageOp)
            and use.operation.parent_op() is op.parent_op()
            and buffer in use.operation.outs
        ]

        if len(in_uses) == 0 or len(out_uses) == 0:
            # only used as input or output, so no risk of a read/write conflict
            # in the index op, we always select this buffer as it is always safe to use.
            new_index = IndexOp(
                index_op.input,
                [*index_op.result_types, buffer.type],
                rewriter.move_region_contents_to_new_regions(index_op.body),
            )
            assert isinstance(yield_op := new_index.body.block.last_op, YieldOp)
            new_yield = YieldOp(*yield_op.operands, buffer)
            rewriter.replace_op(yield_op, new_yield)
            rewriter.replace_op(index_op, new_index, new_results=new_index.results[:-1])

            # replace all uses of the buffer with the result from the index op
            for use in (*in_uses, *out_uses):
                use.operation.operands[use.index] = new_index.results[-1]

            return

        if len(in_uses) != 1 or len(out_uses) != 1:
            raise NotImplementedError("multiple in/out uses of buffer is not yet supported")

        in_use = in_uses[0]
        in_op = cast(StageOp, in_use.operation)
        out_use = out_uses[0]
        out_op = cast(StageOp, out_use.operation)

        if in_op.index.value.data != out_op.index.value.data + 1:
            raise NotImplementedError("non-subsequent in/out uses of buffer is not yet supported")

        if not isinstance(buffer, OpResult) or not isinstance(buffer.op, AllocOp):
            raise NotImplementedError("buffer should be the result of a memref.alloc operation")

        # this is the spot to implement double buffering. There is one in_use, and one out use.
        # we must duplicate the buffer and select the correct one in the index op. to avoid
        # read/write confclits

        # first, we duplicate the buffer:
        buffers = [buffer.op]
        buffers.append(buffer.op.clone())
        rewriter.insert_op(buffers[1], InsertPoint.after(buffer.op))

        # then, we add them to the index op:
        new_index = IndexOp(
            index_op.input,
            [*index_op.result_types, buffer.type],
            rewriter.move_region_contents_to_new_regions(index_op.body),
        )
        assert isinstance(yield_op := new_index.body.block.last_op, YieldOp)

        # here, we insert some logic to select the first buffer on even iterations,
        # and the second buffer on odd iterations
        cst_2 = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())
        cst_0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        rem = arith.RemUIOp(new_index.body.block.args[0], cst_2)
        selection_index = arith.CmpiOp(cst_0, rem, "eq")
        selection = arith.SelectOp(selection_index, buffers[0], buffers[1])
        new_yield = YieldOp(*yield_op.operands, selection)
        rewriter.replace_op(yield_op, [cst_2, cst_0, rem, selection_index, selection, new_yield])
        rewriter.replace_op(index_op, new_index, new_results=new_index.results[:-1])

        # replace all uses of the buffer with the result from the index op
        for use in (*in_uses, *out_uses):
            use.operation.operands[use.index] = new_index.results[-1]

        return


class PipelineDuplicateBuffersPass(ModulePass):
    name = "pipeline-duplicate-buffers"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(PipelineDuplicateBuffers()).rewrite_module(op)
