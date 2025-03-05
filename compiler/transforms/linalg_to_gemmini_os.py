from dataclasses import dataclass, field

from xdsl.dialects import linalg, builtin
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import accfg


from compiler.transforms.insert_accfg_op import InsertAcceleratorOpPattern
from compiler.transforms.convert_linalg_to_accfg import ConnectStatesThroughControlFlowPattern
from compiler.accelerators.gemmini_os import (
        GemminiExAccelerator, 
        GemminiMvinAccelerator, 
        GemminiMvoutAccelerator,
        convert_to_accfg_sequence
    )


@dataclass
class ConvertLinalgToGemminiOsPattern(RewritePattern):
    """
    Specialized conversion pattern that converts any linalg.Generic to
    a set of setup instructions for gemmini.
    """

    module: builtin.ModuleOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter, /):
        if op.library_call is None:
            return

        library_call_name = op.library_call.data
        if library_call_name != "gemmini_os":
            return
        rewriter.replace_matched_op(convert_to_accfg_sequence(op))


class ConvertLinalgToGemminiOsPass(ModulePass):
    name = "convert-linalg-to-gemmini-os"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # insert gemmini_os accelerator ops
        for acc in (GemminiMvinAccelerator(), GemminiMvoutAccelerator(), GemminiExAccelerator()):
            PatternRewriteWalker(InsertAcceleratorOpPattern(acc.generate_acc_op())).rewrite_module(op)
        PatternRewriteWalker(ConvertLinalgToGemminiOsPattern(op)).rewrite_module(op)
        # run these strictly sequentially, otherwise stuff breaks
        PatternRewriteWalker(ConnectStatesThroughControlFlowPattern()).rewrite_module(
            op
        )
