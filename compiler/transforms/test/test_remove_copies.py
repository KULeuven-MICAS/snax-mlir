
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, memref
from xdsl.irdl import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects.test import debug


class RemoveCopies(RewritePattern):
    """Insert debugging function calls"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class RemoveCopiesPass(ModulePass):
    name = "test-remove-copies"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RemoveCopies()).rewrite_module(op)
