from xdsl.dialects import builtin, llvm
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import snax


class ConvertMCycleToLLVM(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, mcycle_op: snax.MCycleOp, rewriter: PatternRewriter):
        """Swap op with call"""
        riscv_mcycle = (
            llvm.InlineAsmOp(
                "csrr $0, mcycle",
                # =r = store result in A 32- or 64-bit
                # general-purpose register (depending on the platform XLEN)
                "=r,~{memory}",
                [],
                [builtin.i32],
                has_side_effects=True,
            ),
        )
        rewriter.replace_matched_op(riscv_mcycle)


class SNAXLowerMCycle(ModulePass):
    name = "snax-lower-mcycle"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertMCycleToLLVM()).rewrite_module(module)
