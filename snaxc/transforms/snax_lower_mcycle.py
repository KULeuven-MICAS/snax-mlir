from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import snax


class ConvertMCycleToLLVM(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, mcycle_op: snax.MCycleOp, rewriter: PatternRewriter):
        """Swap op with call"""
        riscv_mcycle = (
            llvm.InlineAsmOp(
                "csrr zero, mcycle",
                # =r = store result in A 32- or 64-bit
                # general-purpose register (depending on the platform XLEN)
                "~{memory}",
                [],
                [],
                has_side_effects=True,
            ),
        )
        rewriter.insert_op_before_matched_op(riscv_mcycle)
        rewriter.erase_matched_op()


class SNAXLowerMCycle(ModulePass):
    name = "snax-lower-mcycle"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMCycleToLLVM()).rewrite_module(op)
