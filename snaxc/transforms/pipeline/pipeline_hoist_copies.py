from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.memref import CopyOp
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import OpResult
from xdsl.irdl import Operand
from xdsl.parser import IntegerAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


@dataclass
class HoistCopies(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CopyOp, rewriter: PatternRewriter) -> None:
        if not isinstance(for_op := op.parent_op(), ForOp):
            return
        # source changes in this for loop
        if for_op.is_ancestor(op.source.owner):
            return
        # destination changes in this for loop
        if for_op.is_ancestor(op.destination.owner):
            return
        # now decide where to put memref copy:
        source_is_used_in_loop = any(
            use.operation is not op and for_op.is_ancestor(use.operation) for use in op.source.uses
        )
        dest_is_used_in_loop = any(
            use.operation is not op and for_op.is_ancestor(use.operation) for use in op.destination.uses
        )
        if dest_is_used_in_loop:
            assert not source_is_used_in_loop
            op.detach()
            rewriter.insert_op(op, InsertPoint.before(for_op))
        if source_is_used_in_loop:
            assert not dest_is_used_in_loop
            op.detach()
            rewriter.insert_op(op, InsertPoint.after(for_op))


@dataclass(frozen=True)
class PipelineHoistCopies(ModulePass):
    name = "pipeline-hoist-copies"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(HoistCopies()).rewrite_module(op)
