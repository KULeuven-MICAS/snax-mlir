from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.dialects.scf import ForOp
from xdsl.ir import Operation, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects import snax
from snaxc.dialects.pipeline import IndexOp, PipelineOp, StageOp, YieldOp


class UnrollPipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, pipeline: PipelineOp, rewriter: PatternRewriter):
        # check: all stages must have no arguments left
        for operation in pipeline.body.block.ops:
            if isinstance(operation, StageOp) and (
                len(operation.ins) > 0 or len(operation.outs) > 0
            ):
                raise RuntimeError(
                    "found pipeline stage with remaining buffer arguments"
                )

        assert isinstance(for_op := pipeline.parent_op(), ForOp)
        assert isinstance(index_op := pipeline.body.block.first_op, IndexOp)

        # 0: Insert preamble
        index_ops = []
        for i in range(pipeline.nb_stages - 1):
            # create new index op
            index = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
            index_clone = index_op.clone()
            index_clone.operands[0] = index.result
            rewriter.insert_op([index, index_clone], InsertPoint.before(for_op))
            index_ops.append(index_clone)
            # insert stages
            for j in range(i + 1):
                stage = pipeline.stages[j].clone()
                rewriter.insert_op(stage, InsertPoint.before(for_op))
                for operand_0, operand_j in zip(
                    index_op.results, index_ops[i - j].results
                ):
                    operand_0.replace_by_if(
                        operand_j, lambda use: use.operation.parent_op() is stage
                    )
            rewriter.insert_op(snax.ClusterSyncOp(), InsertPoint.before(for_op))

        # 1: Insert postamble (back to front)
        index_ops = []
        index_ops_to_add: list[Operation] = []
        for i in range(pipeline.nb_stages - 1):
            ops_to_add: list[Operation] = []
            # create new index op
            index = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
            index_val = arith.SubiOp(for_op.ub, index)
            index_clone = index_op.clone()
            index_clone.operands[0] = index_val.result
            index_ops_to_add.extend([index, index_val, index_clone])
            index_ops.append(index_clone)
            # insert stages
            for j in reversed(range(i + 1)):
                stage = pipeline.stages[-j - 1].clone()
                for operand_0, operand_j in zip(
                    index_op.results, index_ops[i - j].results
                ):
                    operand_0.replace_by_if(
                        operand_j, lambda use: use.operation.parent_op() is stage
                    )
                ops_to_add.append(stage)
            ops_to_add.append(snax.ClusterSyncOp())
            rewriter.insert_op(ops_to_add, InsertPoint.after(for_op))
        rewriter.insert_op(index_ops_to_add, InsertPoint.after(for_op))

        # 2: Increment for op lower bound by (nb_stages - 1)
        cst = arith.ConstantOp.from_int_and_width(
            pipeline.nb_stages - 1, builtin.IndexType()
        )
        for_op.lb.replace_by_if(cst.result, lambda use: use.operation is for_op)
        rewriter.insert_op(cst, InsertPoint.before(for_op))

        # 3: Create copies of the index op with other index values
        index_ops: list[IndexOp] = [index_op]
        for i in range(1, pipeline.nb_stages):
            cst = arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
            index_val = arith.SubiOp(index_op.input, cst)
            clone = index_op.clone()
            # set new input
            clone.operands[0] = index_val.result
            # insert ops and add to list
            rewriter.insert_op(
                [cst, index_val, clone], InsertPoint.at_start(pipeline.body.block)
            )
            # replace usage by new index op
            for operand_0, operand_i in zip(index_op.results, clone.results):

                def belongs_to_stage_i(use: Use):
                    assert isinstance(stage := use.operation.parent_op(), StageOp)
                    return stage.index.value.data == i

                operand_0.replace_by_if(operand_i, belongs_to_stage_i)
            index_ops.append(clone)

        # 4: Add synchronization at end of pipeline
        rewriter.insert_op(
            snax.ClusterSyncOp(), InsertPoint.at_end(pipeline.body.block)
        )

        # 5: Inline region from pipeline op
        rewriter.inline_block(pipeline.body.block, InsertPoint.before(pipeline))
        rewriter.erase_op(pipeline)


class DestructIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexOp, rewriter: PatternRewriter):
        # should no longer belong to a pipeline:
        if isinstance(op.parent_op(), PipelineOp):
            return

        # replace block arg by index op input
        op.body.block.args[0].replace_by(op.input)

        # replace op results by yield values
        assert isinstance(yield_op := op.body.block.last_op, YieldOp)
        for result, yield_value in zip(op.results, yield_op.arguments):
            result.replace_by(yield_value)

        # remove yield op and inline index ops
        rewriter.erase_op(yield_op)
        rewriter.inline_block(op.body.block, InsertPoint.before(op))
        rewriter.erase_op(op)


class DestructStage(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StageOp, rewriter: PatternRewriter):
        # should no longer belong to a pipeline:
        if isinstance(op.parent_op(), PipelineOp):
            return

        # should have 0 operands:
        if len(op.operands) != 0:
            raise RuntimeError("found stage with remaining buffer arguments")

        # inline stage
        rewriter.inline_block(op.body.block, InsertPoint.before(op))
        rewriter.erase_op(op)


@dataclass(frozen=True)
class UnrollPipelinePass(ModulePass):
    name = "unroll-pipeline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(UnrollPipeline()).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier([DestructIndex(), DestructStage()])
        ).rewrite_module(op)
