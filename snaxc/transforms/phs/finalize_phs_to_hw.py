from xdsl.context import Context
from xdsl.dialects import builtin, comb, hw
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.reconcile_unrealized_casts import reconcile_unrealized_casts

from snaxc.dialects import phs
from snaxc.phs.hw_conversion import get_choice_bitwidth


class ConvertMuxes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, mux: phs.MuxOp, rewriter: PatternRewriter):
        # 0 = lhs, 1 = rhs
        # Change type of mux switch
        cast_op, res = builtin.UnrealizedConversionCastOp.cast_one(mux.switch, builtin.IntegerType(1))
        rewriter.insert_op(cast_op)
        new_mux = comb.MuxOp(res, mux.rhs, mux.lhs)
        rewriter.replace_op(mux, [new_mux])


class ConvertChooseOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, choose_op: phs.ChooseOp, rewriter: PatternRewriter):
        index_bw = get_choice_bitwidth(choose_op)
        yield_results: list[SSAValue] = []
        for region in choose_op.regions:
            (parent_block,) = (choose_op.parent_block(),)
            assert parent_block is not None
            yield_op = region.block.ops.last
            assert isinstance(yield_op, phs.YieldOp)
            if not len(yield_op.operands) == 1:
                raise NotImplementedError
            yield_results.append(yield_op.operands[0])
            ip = InsertPoint(parent_block, insert_before=choose_op)
            rewriter.inline_block(insertion_point=ip, block=region.block, arg_values=choose_op.data_operands)
            yield_op.detach()
            rewriter.erase_op(yield_op)

        # SystemVerilog needs reversed operand ordering for array indexing
        rewriter.insert_op(create_array := hw.ArrayCreateOp(*reversed(yield_results)))
        cast_op, res = builtin.UnrealizedConversionCastOp.cast_one(choose_op.switch, builtin.IntegerType(index_bw))
        rewriter.insert_op(cast_op)
        rewriter.replace_op(choose_op, hw.ArrayGetOp(create_array, res))


class FinalizePhsToHWPass(ModulePass):
    name = "finalize-phs-to-hw"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMuxes(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertChooseOps(), apply_recursively=False).rewrite_module(op)
        reconcile_unrealized_casts(op)
