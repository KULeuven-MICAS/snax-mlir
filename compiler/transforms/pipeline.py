from inspect import BlockFinder

from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg, memref, scf
from xdsl.dialects.arith import Addi, Constant, Muli, Subi
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    MemRefType,
    NoneAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.irdl import Operand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects import snax, stream
from compiler.dialects.pipeline import DoubleStageOp, PipelineOp, StageOp
from compiler.dialects.tsl import TiledStridedLayoutAttr
from compiler.util.snax_memory import L1


class ConstructPipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter):
        if isinstance(op.body.block.first_op, scf.For):
            return
        # create pipeline op inside
        pipeline_op = PipelineOp(Region(Block()))

        next_op = op.body.block.first_op
        assert next_op is not None

        input_buffers = []
        output_buffers = []
        ops_to_move: list[Operation] = []

        while True:
            if isinstance(next_op, memref.CopyOp):
                input_buffers.append(next_op.source)
                output_buffers.append(next_op.destination)

            if isinstance(next_op, linalg.Generic):
                input_buffers.extend([o for o in next_op.inputs if isinstance(o.type, MemRefType)])
                output_buffers.extend([o for o in next_op.outputs if isinstance(o.type, MemRefType)])

            if isinstance(next_op, stream.StreamingRegionOp):
                input_buffers.extend([o for o in next_op.inputs if isinstance(o.type, MemRefType)])
                output_buffers.extend([o for o in next_op.outputs if isinstance(o.type, MemRefType)])

            ops_to_move.append(next_op)

            assert (next_op := next_op.next_op) is not None

            if isinstance(next_op, snax.ClusterSyncOp | scf.Yield):
                # handle layout cast problem
                for i, input_buffer in enumerate(input_buffers):
                    if (
                        isinstance(input_buffer, OpResult)
                        and isinstance(input_buffer.op, snax.LayoutCast)
                        and input_buffer.op in ops_to_move
                    ):
                        input_buffers[i] = input_buffer.op.source

                for i, output_buffer in enumerate(output_buffers):
                    if (
                        isinstance(output_buffer, OpResult)
                        and isinstance(output_buffer.op, snax.LayoutCast)
                        and output_buffer.op in ops_to_move
                    ):
                        output_buffers[i] = output_buffer.op.source

                # create new pipeline stage
                stage = StageOp(
                    input_buffers,
                    output_buffers,
                    body=Region(Block(arg_types=[o.type for o in (*input_buffers, *output_buffers)])),
                )

                operand_to_block_args = {
                    operand: block_arg
                    for operand, block_arg in zip((*input_buffers, *output_buffers), stage.body.block.args)
                }

                for op_to_move in ops_to_move:
                    op_to_move.detach()
                    for i, operand in enumerate(op_to_move.operands):
                        if operand in operand_to_block_args:
                            op_to_move.operands[i] = operand_to_block_args[operand]
                    rewriter.insert_op(op_to_move, InsertPoint.at_end(stage.body.block))

                rewriter.insert_op(stage, InsertPoint.at_end(pipeline_op.body.block))

                # if scf yield, end of for loop, exit pass
                if isinstance(next_op, scf.Yield):
                    break

                # else, drop sync op and move on to next stage
                sync_op = next_op
                assert (next_op := next_op.next_op) is not None
                rewriter.erase_op(sync_op)

                input_buffers = []
                output_buffers = []
                ops_to_move: list[Operation] = []

        # insert pipeline op
        rewriter.insert_op(pipeline_op, InsertPoint.at_start(op.body.block))


class DuplicateBuffers(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        # first, duplicate all buffers

        duplicated_buffers: dict[SSAValue, SSAValue] = {}

        for stage in op.body.block.ops:
            assert isinstance(stage, StageOp)
            # duplicate output buffers
            for output_buffer in stage.output_buffers:
                if isinstance(output_buffer, OpResult) and isinstance(output_buffer.op, memref.Alloc):
                    cloned = output_buffer.op.clone()
                    rewriter.insert_op(cloned, InsertPoint.after(output_buffer.op))
                    duplicated_buffers[output_buffer.op.memref] = cloned.memref

        for stage in op.body.block.ops:
            assert isinstance(stage, StageOp)
            # duplicate collapse shape ops
            for input_buffer in stage.input_buffers:
                if isinstance(input_buffer, OpResult) and isinstance(input_buffer.op, memref.CollapseShapeOp):
                    cloned = input_buffer.op.clone()
                    cloned.operands[0] = duplicated_buffers[cloned.operands[0]]
                    rewriter.insert_op(cloned, InsertPoint.after(input_buffer.op))
                    duplicated_buffers[input_buffer.op.result] = cloned.result

        for stage in op.body.block.ops:
            assert isinstance(stage, StageOp)
            # create mirrors for buffers that need no duplication
            for buffer in (*stage.input_buffers, *stage.output_buffers):
                if buffer not in duplicated_buffers:
                    duplicated_buffers[buffer] = buffer

        # now, turn all pipeline stages into double stages
        even = True
        for stage in op.body.block.ops:
            assert isinstance(stage, StageOp)
            if even:
                double_op = DoubleStageOp(
                    input_buffers_even=stage.input_buffers,
                    input_buffers_odd=[duplicated_buffers[o] for o in stage.input_buffers],
                    output_buffers_even=stage.output_buffers,
                    output_buffers_odd=[duplicated_buffers[o] for o in stage.output_buffers],
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
                )
            else:
                double_op = DoubleStageOp(
                    input_buffers_even=[duplicated_buffers[o] for o in stage.input_buffers],
                    input_buffers_odd=stage.input_buffers,
                    output_buffers_even=[duplicated_buffers[o] for o in stage.output_buffers],
                    output_buffers_odd=stage.output_buffers,
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
                )
            rewriter.replace_op(stage, double_op)
            even = not even


class UnrollPipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        if "unrolled" in op.attributes:
            return

        # double the pipeline op
        rewriter.insert_op(clone := op.clone(), InsertPoint.after(op))

        op.attributes["unrolled"] = StringAttr("even")
        clone.attributes["unrolled"] = StringAttr("odd")

        # insert sync after every stage
        rewriter.insert_op(snax.ClusterSyncOp(), InsertPoint.after(op))
        rewriter.insert_op(snax.ClusterSyncOp(), InsertPoint.after(clone))


class UndoubleStages(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        if "unrolled" not in op.attributes:
            return
        assert isinstance(even_ness := op.attributes["unrolled"], StringAttr)
        even = even_ness.data == 'even'

        for stage in op.body.block.ops:
            assert isinstance(stage, DoubleStageOp)
            if even:
                stage_op = StageOp(
                    input_buffers=stage.input_buffers_even,
                    output_buffers=stage.output_buffers_even,
                    body=rewriter.move_region_contents_to_new_regions(stage.body)
                )
            else:
                stage_op = StageOp(
                    input_buffers=stage.input_buffers_odd,
                    output_buffers=stage.output_buffers_odd,
                    body=rewriter.move_region_contents_to_new_regions(stage.body)
                )
            rewriter.replace_op(stage, stage_op)

        del op.attributes["unrolled"]


class DestructStages(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StageOp, rewriter: PatternRewriter):

        # replace all block args with op operands
        for block_arg, operand in zip(op.body.block.args, op.operands):
            block_arg.replace_by(operand)

        # replace the op by all its operations
        operations_in_stage = []
        for op_in_stage in op.body.block.ops:
            op_in_stage.detach()
            operations_in_stage.append(op_in_stage)
        rewriter.replace_matched_op(operations_in_stage)


class RemovePipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):

        operations_in_stage = []
        for op_in_stage in op.body.block.ops:
            op_in_stage.detach()
            operations_in_stage.append(op_in_stage)
        rewriter.replace_matched_op(operations_in_stage)


class PipelinePass(ModulePass):
    name = "pipeline"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConstructPipeline(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(DuplicateBuffers(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(UnrollPipeline(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(UndoubleStages(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(DestructStages(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RemovePipeline(), apply_recursively=False).rewrite_module(op)
