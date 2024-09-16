from xdsl.context import MLContext
from xdsl.dialects import builtin, linalg
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects.kernel import Kernel, Parsable


def check_block(block_a: Block, block_b: Block) -> bool:
    """
    Verify if two blocks are equal to each other,
    that for the same inputs they include the same
    operations.
    """
    if len(block_a.ops) != len(block_b.ops):
        return False

    # warning: this is a bit of a naive way of checking equality between
    # kernels, but should cover all of our purposes for quite some time
    for op_a, op_b in zip(block_a.ops, block_b.ops, strict=True):
        if type(op_a) is not type(op_b):
            return False

    return True


class ParseLinalgBody(RewritePattern):
    """
    Matches on linalg.generic operations to check if
    their body matches a specific kernel op defined in
    the kernel dialect. Replaces the body with the relevant
    kernel op if this is true.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.Generic, rewriter: PatternRewriter):
        for op_def in Kernel.operations:
            if not issubclass(op_def, Parsable):
                continue
            try:
                kernel_op = op_def(
                    operands=linalg_op.body.block.args[:-1],
                    result_types=[linalg_op.body.block.args[-1].type],
                )
            except ValueError:
                # wrong number of operands for this operation, no match
                continue

            if check_block(linalg_op.body.block, kernel_op.parsing_region.block):
                # modify linalg body
                # delete all previous ops:
                while linalg_op.body.block.last_op:
                    rewriter.erase_op(linalg_op.body.block.last_op)

                # insert new kernel op kernel in body
                rewriter.insert_op(
                    (kernel_op, linalg.YieldOp(kernel_op)),
                    InsertPoint.at_end(linalg_op.body.block),
                )


class ConvertLinalgToKernel(ModulePass):
    name = "convert-linalg-to-kernel"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ParseLinalgBody()).rewrite_module(op)
