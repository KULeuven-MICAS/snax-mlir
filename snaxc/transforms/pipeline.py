from typing import Sequence
from xdsl.context import Context
from xdsl.dialects import builtin, linalg, memref, scf
from xdsl.dialects import arith
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    StringAttr,
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

from snaxc.dialects import dart, snax
from snaxc.dialects.pipeline import DoubleStageOp, IndexOp, PipelineOp, StageOp, YieldOp
from snaxc.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm

# helper function to extract the value from an operand known to be a constant
def extract_constant(operand: Operand) -> int:
    assert isinstance(operand, OpResult)
    assert isinstance(const_op := operand.op, ConstantOp)
    assert isinstance(const_val := const_op.value, IntegerAttr)
    return const_val.value.data

class CombineForLoops(RewritePattern):
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter):
        # find nested for loops
        parent_op = op

        # first op should for op and only op in the block
        if not isinstance(child_op := op.body.block.first_op, scf.ForOp):
            return
        if not isinstance(child_op.next_op, scf.YieldOp):
            return

        # should start from c0
        assert extract_constant(parent_op.lb) == 0
        assert extract_constant(child_op.lb) == 0

        # get steps with
        parent_step = extract_constant(parent_op.step)
        child_step = extract_constant(child_op.step)

        # get upper bounds
        parent_ub = extract_constant(parent_op.ub)
        child_ub = extract_constant(child_op.ub)

        full_range = parent_ub // parent_step * child_ub // child_step

        new_lb = ConstantOp(IntegerAttr.from_index_int_value(0))
        new_step = ConstantOp(IntegerAttr.from_index_int_value(1))
        new_ub = ConstantOp(IntegerAttr.from_index_int_value(full_range))
        child_ub = ConstantOp(IntegerAttr.from_index_int_value(child_ub // child_step))

        # create a new for loop op
        new_for_region = Region(new_for_block := Block([], arg_types=[IndexType()]))
        new_for = scf.ForOp(new_lb, new_ub, new_step, [], new_for_region)

        # compute new iteration variables
        parent_iter = arith.DivUIOp(new_for_block.args[0], child_ub)
        parent_iter_val = arith.MuliOp(parent_iter.result, parent_op.step)
        parent_op.body.block.args[0].replace_by(parent_iter_val.result)

        child_iter = arith.RemUIOp(new_for_block.args[0], child_ub)
        child_iter_val = arith.MuliOp(child_iter.result, child_op.step)
        child_op.body.block.args[0].replace_by(child_iter_val.result)

        # populate new for loop block
        rewriter.insert_op((parent_iter, parent_iter_val, child_iter, child_iter_val), InsertPoint.at_start(new_for_block))
        for inner_op in child_op.body.block.ops:
            inner_op.detach()
            rewriter.insert_op(inner_op, InsertPoint.at_end(new_for_block))

        # replace the original for loop operation
        rewriter.replace_matched_op((new_lb, new_step, new_ub, child_ub, new_for))


class ConstructPipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter):
        if isinstance(op.body.block.first_op, scf.ForOp):
            return
        # create pipeline op inside
        pipeline_op = PipelineOp(Region(Block()))

        # create pipeline index op
        index_op = IndexOp.empty(op.body.block.args[0])

        next_op = op.body.block.first_op
        assert next_op is not None

        # first get all indexing ops
        # these should be at the beginning of the for loop
        # this tends to be true, but we could canonicalize 
        # this case in case it isn't for some reason
        while True:
            # TODO: this check should improve in the future
            if dispatch_to_compute(next_op) or dispatch_to_dm(next_op):
                break
            op_to_move = next_op
            assert (next_op := next_op.next_op) is not None
            op_to_move.detach()
            index_op = index_op.add_op(op_to_move, rewriter)


        # canonicalize index op
        index_op = index_op.clear_unused_args(rewriter)

        # insert index op
        rewriter.insert_op(index_op, InsertPoint.at_end(pipeline_op.body.block))

        input_buffers: list[SSAValue] = []
        output_buffers: list[SSAValue] = []
        ops_to_move: list[Operation] = []

        while True:

            if isinstance(next_op, memref.CopyOp):
                input_buffers.append(next_op.source)
                output_buffers.append(next_op.destination)

            if isinstance(next_op, linalg.GenericOp):
                input_buffers.extend(
                    [o for o in next_op.inputs if isinstance(o.type, MemRefType)]
                )
                output_buffers.extend(
                    [o for o in next_op.outputs if isinstance(o.type, MemRefType)]
                )

            if isinstance(next_op, dart.StreamingRegionOpBase):
                input_buffers.extend(
                    [o for o in next_op.inputs if isinstance(o.type, MemRefType)]
                )
                output_buffers.extend(
                    [o for o in next_op.outputs if isinstance(o.type, MemRefType)]
                )

            ops_to_move.append(next_op)

            assert (next_op := next_op.next_op) is not None

            if isinstance(next_op, snax.ClusterSyncOp | scf.YieldOp):

                # create new pipeline stage
                stage = StageOp(
                    input_buffers,
                    output_buffers,
                    body=Region(
                        Block(
                            arg_types=[
                                o.type for o in (*input_buffers, *output_buffers)
                            ]
                        )
                    ),
                )

                operand_to_block_args = {
                    operand: block_arg
                    for operand, block_arg in zip(
                        (*input_buffers, *output_buffers), stage.body.block.args
                    )
                }

                for op_to_move in ops_to_move:
                    op_to_move.detach()
                    for i, operand in enumerate(op_to_move.operands):
                        if operand in operand_to_block_args:
                            op_to_move.operands[i] = operand_to_block_args[operand]
                    rewriter.insert_op(op_to_move, InsertPoint.at_end(stage.body.block))

                rewriter.insert_op(stage, InsertPoint.at_end(pipeline_op.body.block))

                # if scf yield, end of for loop, exit pass
                if isinstance(next_op, scf.YieldOp):
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
            if not isinstance(stage, StageOp):
                continue
            # duplicate output buffers
            for output_buffer in stage.output_buffers:
                if isinstance(output_buffer, OpResult) and isinstance(
                    output_buffer.op, memref.AllocOp
                ):
                    cloned = output_buffer.op.clone()
                    rewriter.insert_op(cloned, InsertPoint.after(output_buffer.op))
                    duplicated_buffers[output_buffer.op.memref] = cloned.memref

        for stage in op.body.block.ops:
            if not isinstance(stage, StageOp):
                continue
            # duplicate collapse shape ops
            for input_buffer in stage.input_buffers:
                if isinstance(input_buffer, OpResult) and isinstance(
                    input_buffer.op, memref.CollapseShapeOp
                ):
                    cloned = input_buffer.op.clone()
                    cloned.operands[0] = duplicated_buffers[cloned.operands[0]]
                    rewriter.insert_op(cloned, InsertPoint.after(input_buffer.op))
                    duplicated_buffers[input_buffer.op.result] = cloned.result

        for stage in op.body.block.ops:
            if not isinstance(stage, StageOp):
                continue
            # create mirrors for buffers that need no duplication
            for buffer in (*stage.input_buffers, *stage.output_buffers):
                if buffer not in duplicated_buffers:
                    duplicated_buffers[buffer] = buffer

        # now, turn all pipeline stages into double stages
        even = True
        for stage in op.body.block.ops:
            if not isinstance(stage, StageOp):
                continue
            if even:
                double_op = DoubleStageOp(
                    input_buffers_even=stage.input_buffers,
                    input_buffers_odd=[
                        duplicated_buffers[o] for o in stage.input_buffers
                    ],
                    output_buffers_even=stage.output_buffers,
                    output_buffers_odd=[
                        duplicated_buffers[o] for o in stage.output_buffers
                    ],
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
                )
            else:
                double_op = DoubleStageOp(
                    input_buffers_even=[
                        duplicated_buffers[o] for o in stage.input_buffers
                    ],
                    input_buffers_odd=stage.input_buffers,
                    output_buffers_even=[
                        duplicated_buffers[o] for o in stage.output_buffers
                    ],
                    output_buffers_odd=stage.output_buffers,
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
                )
            rewriter.replace_op(stage, double_op)
            even = not even


def get_single_stage(double_stage: DoubleStageOp, even: bool, rewriter: PatternRewriter):
    if even:
        stage_op = StageOp(
            input_buffers=double_stage.input_buffers_even,
            output_buffers=double_stage.output_buffers_even,
            body=rewriter.move_region_contents_to_new_regions(double_stage.body),
        )
    else:
        stage_op = StageOp(
            input_buffers=double_stage.input_buffers_odd,
            output_buffers=double_stage.output_buffers_odd,
            body=rewriter.move_region_contents_to_new_regions(double_stage.body),
        )
    return stage_op
 

class UnrollPipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        if "unrolled" in op.attributes:
            return

        for_op = op.parent_op()
        assert isinstance(for_op, scf.ForOp)

        # start assumed to be 0, step assumed to be 1
        ub = extract_constant(for_op.ub)

        # TODO: handle case for uneven nb of iterations
        if ub % 2 != 0:
            raise NotImplementedError("Uneven number of iterations is currently unsupported")

        # get the index operation
        index_op = op.body.block.first_op
        assert isinstance(index_op, IndexOp)

        # get the double stages
        stages: list[DoubleStageOp] = [stage for stage in op.body.block.ops if isinstance(stage, DoubleStageOp)]

        # insert preambles
        # for 2 stages, need a single preamble of 1 stage
        # for 3 stages, need two preambles of 1 stage and 2 stages
        # ...

        preambles: list[Operation] = []

        for i in range(len(stages) - 1):

            # add preamble stages
            for j in range(i + 1):

                # create index op with cst reference
                cst = ConstantOp(IntegerAttr.from_index_int_value(i - j))
                preamble_index = index_op.clone()
                preamble_index.operands[0] = cst.result

                stage = stages[j].clone()

                # turn into single stages
                single_stage = get_single_stage(stage, i % 2 == 0, rewriter)
                stage.erase()

                # use preamble index instead
                for k, operand in enumerate(single_stage.operands):
                    if isinstance(operand, OpResult) and operand.op is index_op:
                        single_stage.operands[k] = preamble_index.results[operand.index]

                preambles.extend([cst, preamble_index, single_stage])

            # add synchronization
            preambles.extend([snax.ClusterSyncOp()])

        # insert the preambles
        rewriter.insert_op(preambles, InsertPoint.before(for_op))

        # restructure the for loop, assume nb_iters is large enough to fully unroll

        new_for_loop_ops: list[Operation]
        # create n + 1 index ops
        for i in range(len(stages) + 1):
            pass

        # double the pipeline op
        rewriter.insert_op(clone := op.clone(), InsertPoint.after(op))

        op.attributes["unrolled"] = StringAttr("even")
        clone.attributes["unrolled"] = StringAttr("odd")

        # insert sync after every stage
        rewriter.insert_op(snax.ClusterSyncOp(), InsertPoint.after(op))
        rewriter.insert_op(snax.ClusterSyncOp(), InsertPoint.after(clone))

        # set start
        new_lb = ConstantOp(IntegerAttr.from_index_int_value(len(stages) - 1))
        new_step = ConstantOp(IntegerAttr.from_index_int_value(2))
        new_ub = ConstantOp(IntegerAttr.from_index_int_value(ub))

        # replace by for loop with new iterations
        new_for = scf.ForOp(new_lb, new_ub, new_step, [], rewriter.move_region_contents_to_new_regions(for_op.body))
        rewriter.replace_op(for_op, [new_lb, new_ub, new_step, new_for])

        # insert the postambles
        # for 2 stages, need a single postamble of 1 stage
        # for 3 stages, need two postambles of 2 stages and 1 stage
        # ...

        postambles: list[Operation] = []

        for i in reversed(range(len(stages) - 1)):

            # add postamble stages
            for j in reversed(range(i + 1)):

                # create index op with cst reference
                cst = ConstantOp(IntegerAttr.from_index_int_value(ub - (i - j) - 1))
                postamble_index = index_op.clone()
                postamble_index.operands[0] = cst.result

                stage = stages[-j-1].clone()

                # turn into single stages 
                # TODO: figure out the correct even/odd thing here
                single_stage = get_single_stage(stage, i % 2 == 1, rewriter)
                stage.erase()

                # use postamble index instead
                for k, operand in enumerate(single_stage.operands):
                    if isinstance(operand, OpResult) and operand.op is index_op:
                        single_stage.operands[k] = postamble_index.results[operand.index]

                postambles.extend([cst, postamble_index, single_stage])

            # add synchronization
            postambles.extend([snax.ClusterSyncOp()])

        # insert the preambles
        rewriter.insert_op(postambles, InsertPoint.after(new_for))




class UndoubleStages(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        if "unrolled" not in op.attributes:
            return
        assert isinstance(even_ness := op.attributes["unrolled"], StringAttr)
        even = even_ness.data == "even"

        for stage in op.body.block.ops:
            if not isinstance(stage, DoubleStageOp):
                continue
            if even:
                stage_op = StageOp(
                    input_buffers=stage.input_buffers_even,
                    output_buffers=stage.output_buffers_even,
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
                )
            else:
                stage_op = StageOp(
                    input_buffers=stage.input_buffers_odd,
                    output_buffers=stage.output_buffers_odd,
                    body=rewriter.move_region_contents_to_new_regions(stage.body),
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
        operations_in_stage: list[Operation] = []
        for op_in_stage in op.body.block.ops:
            op_in_stage.detach()
            operations_in_stage.append(op_in_stage)
        rewriter.replace_matched_op(operations_in_stage)


class RemovePipeline(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PipelineOp, rewriter: PatternRewriter):
        operations_in_stage: list[Operation] = []
        for op_in_stage in op.body.block.ops:
            op_in_stage.detach()
            operations_in_stage.append(op_in_stage)
        rewriter.replace_matched_op(operations_in_stage)


class RemovePipelineIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexOp, rewriter: PatternRewriter):
        operations_in_stage: list[Operation] = []
        new_results: Sequence[SSAValue] = []
        for op_in_stage in op.body.block.ops:
            if isinstance(op_in_stage, YieldOp):
                new_results.extend([arg for arg in op_in_stage.arguments])
            else:
                op_in_stage.detach()
                for i, operand in enumerate(op_in_stage.operands):
                    if operand is op.body.block.args[0]:
                        op_in_stage.operands[i] = op.input
                operations_in_stage.append(op_in_stage)
        rewriter.replace_matched_op(operations_in_stage, new_results)

class PipelinePass(ModulePass):
    name = "pipeline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(CombineForLoops()).rewrite_module(op)
        PatternRewriteWalker(ConstructPipeline(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(DuplicateBuffers(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(UnrollPipeline(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(UndoubleStages(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(DestructStages(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RemovePipeline(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RemovePipelineIndex(), apply_recursively=False).rewrite_module(op)
