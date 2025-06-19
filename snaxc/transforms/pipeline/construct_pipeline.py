from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, scf
from xdsl.dialects.builtin import IndexType, MemRefType
from xdsl.dialects.linalg import GenericOp
from xdsl.dialects.memref import CopyOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import Operand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects.dart import StreamingRegionOpBase
from snaxc.dialects.pipeline import IndexOp, PipelineOp, StageOp, YieldOp
from snaxc.dialects.snax import ClusterSyncOp
from snaxc.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm


@dataclass
class ConstructPipeline(RewritePattern):
    """
    This pattern will construct a pipeline operation from a for loop.
    It tries to detect a structure in the following manner:
    ```
    for:
        (index_ops):
        # some ops doing basic math or taking memref subviews
        # generally no side-effects here
        (stage_1):
        # some dispatchable op (memref copy, generic, streaming region)
        (cluster_sync)
        (stage_2)
        (cluster_sync)
        ...
    ```
    This structure is then converted to a pipeline op, such that it can
    be urnolled in further transformations to achieve double buffering
    and pipelined execution of asynchronous multi-core accelerators.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ForOp, rewriter: PatternRewriter):
        # TODO: only apply for for loops with lb 0 and step 1

        # no nested for loop allowed
        for operation in op.walk():
            if operation is not op and isinstance(operation, ForOp):
                return

        # create pipeline op inside
        pipeline_op = PipelineOp(Region(Block()))

        # create pipeline index op with for loop iter value as index
        index_ops: Sequence[Operation] = []

        next_op: Operation | None = op.body.block.first_op
        assert next_op is not None

        def is_index_op(op: Operation) -> bool:
            if dispatch_to_compute(op) or dispatch_to_dm(op):
                return False
            if isinstance(op, ClusterSyncOp):
                return False
            return True

        # first get all indexing ops
        while is_index_op(next_op):
            index_ops.append(next_op)
            if isinstance(next_op := next_op.next_op, scf.YieldOp):
                # no valid pipeline detected
                return
            assert next_op is not None

        # now fetch the stages
        stages: list[list[Operation]] = []
        current_stage: list[Operation] = []
        cluster_sync_ops: list[ClusterSyncOp] = []

        def is_stage_op(op: Operation) -> bool:
            return isinstance(op, CopyOp | GenericOp | StreamingRegionOpBase)

        while is_stage_op(next_op):
            current_stage.append(next_op)
            next_op = next_op.next_op
            if isinstance(next_op, ClusterSyncOp):
                stages.append(current_stage)
                current_stage = []
                # save cluster sync
                cluster_sync_ops.append(next_op)
                next_op = next_op.next_op
            if isinstance(next_op, scf.YieldOp):
                # valid pipeline ends with sync, so current
                # stage should be empty
                if len(current_stage) > 0:
                    return
                break
            assert next_op is not None

        # a valid pipeline has at least two stages
        if len(stages) < 2:
            return

        # at this point, the correct pipeline is detected, now we should create the
        # operations for it

        pipeline_op = PipelineOp(Region(Block()))

        # create index op
        index_args = [i_op.results[0] for i_op in index_ops if len(i_op.results) > 0]
        index_yield = YieldOp(*index_args)
        for o in index_ops:
            o.detach()
        index_op = IndexOp(
            input=op.body.block.args[0],
            result_types=[x.type for x in index_args],
            body=Region(Block([*index_ops, index_yield], arg_types=[IndexType()])),
        )
        # replace uses of index with the index block arg
        index_op.input.replace_by_if(index_op.body.block.args[0], lambda use: use.operation in index_ops)

        # insert index op
        rewriter.insert_op(index_op, InsertPoint.at_end(pipeline_op.body.block))

        # create stages
        for i, stage in enumerate(stages):
            input_buffers: list[SSAValue] = []
            output_buffers: list[SSAValue] = []

            stage_block = Block([])

            for operation in stage:
                operation.detach()

                def rewrite_operand(operand: Operand, index: int, is_input: bool):
                    if is_input:
                        input_buffers.append(operand)
                        arg_insert_index = len(input_buffers) - 1
                    else:
                        output_buffers.append(operand)
                        arg_insert_index = len(input_buffers) + len(output_buffers) - 1
                    operation.operands[index] = stage_block.insert_arg(operand.type, arg_insert_index)

                if isinstance(operation, CopyOp):
                    rewrite_operand(operation.source, 0, True)
                    rewrite_operand(operation.destination, 1, False)
                elif isinstance(operation, GenericOp):
                    for i, operand in enumerate(operation.operands):
                        if isinstance(operand.type, MemRefType):
                            rewrite_operand(operand, i, operand in operation.inputs)
                elif isinstance(operation, StreamingRegionOpBase):
                    for i, operand in enumerate(operation.operands):
                        if isinstance(operand.type, MemRefType):
                            rewrite_operand(operand, i, operand in operation.inputs)

                stage_block.add_op(operation)

            stage_op = StageOp(
                input_buffers,
                output_buffers,
                i,
                body=Region(stage_block),
            )

            rewriter.insert_op(stage_op, InsertPoint.at_end(pipeline_op.body.block))

        # insert pipeline op
        rewriter.insert_op(pipeline_op, InsertPoint.at_start(op.body.block))

        # remove the remaining cluster syncs
        for cluster_sync in cluster_sync_ops:
            rewriter.erase_op(cluster_sync)


class ConstructPipelinePass(ModulePass):
    name = "construct-pipeline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConstructPipeline(), apply_recursively=False).rewrite_module(op)
