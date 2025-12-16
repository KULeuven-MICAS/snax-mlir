from xdsl.context import Context
from xdsl.dialects import builtin, comb, hw
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.mlir_opt import MLIROptPass

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
        yield_results: list[SSAValue] = []
        for region in choose_op.regions:
            for op in region.ops:
                # Move all non-yield operations outside the choice block
                if not isinstance(op, phs.YieldOp):
                    op.detach()
                    rewriter.insert_op(op)
                # put all yielded results in one big array
                else:
                    if not len(list(op.operands)) == 1:
                        raise NotImplementedError()
                    for operand in op.operands:
                        assert not isinstance(operand, Block)
                        yield_results.append(operand)
        rewriter.insert_op(create_array := hw.ArrayCreateOp(*yield_results))
        index_bw = get_choice_bitwidth(choose_op)
        cast_op, res = builtin.UnrealizedConversionCastOp.cast_one(choose_op.switch, builtin.IntegerType(index_bw))
        rewriter.insert_op(cast_op)
        rewriter.replace_op(choose_op, hw.ArrayGetOp(create_array, res))


class FinalizePhsToHWPass(ModulePass):
    name = "finalize-phs-to-hw"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMuxes(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertChooseOps(), apply_recursively=False).rewrite_module(op)
        MLIROptPass(
            executable="circt-opt",
            generic=True,
            arguments=("--reconcile-unrealized-casts", "--allow-unregistered-dialect"),
        ).apply(ctx, op)
        MLIROptPass(
            executable="circt-opt", generic=True, arguments=("--map-arith-to-comb", "--allow-unregistered-dialect")
        ).apply(ctx, op)
