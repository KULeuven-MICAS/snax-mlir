from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin, memref, scf
from xdsl.dialects.builtin import MemRefType
from xdsl.dialects.func import Return
from xdsl.ir.core import Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.mlir_opt import MLIROptPass

from compiler.util.snax_memory import L1


class ClearIterArgs(RewritePattern):
    """
    Warning: do not use this pass, super hacky, but I don't know
    how to handle this otherwise for now...
    I don't know if the bufferized loop is correct. seems fishy
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter):
        # remove all iter args, although we should probably check if they are still used
        if len(op.iter_args) == 0:
            return

        new_results = []

        for result, for_result in zip(op.results, op.body.block.last_op.operands):
            result.replace_by(for_result)
            new_results.append(for_result)

        rewriter.replace_op(op.body.block.last_op, scf.Yield())

        rewriter.replace_matched_op(
            new_op := scf.For(op.lb, op.ub, op.step, [], rewriter.move_region_contents_to_new_regions(op.body)), new_results
        )

        new_op.body.block._args = new_op.body.block._args[:1]


@dataclass(frozen=True)
class SnaxBufferize(ModulePass):
    name = "snax-bufferize"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)

    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops"
            + " function-boundary-type-conversion=identity-layout-map",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(ClearIterArgs()).rewrite_module(op)
