from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.ir import Block
from xdsl.parser import IRDLOperation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects.kernel import Kernel, Parsable


def check_kernel_equivalence(block_a: Block, block_b: Block) -> bool:
    """
    Verify if two blocks are equivalent to each other,
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
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        for op_def in Kernel.operations:
            if not issubclass(op_def, Parsable):
                # not a parsable op, continue search
                continue
            assert issubclass(op_def, IRDLOperation)
            if len(op_def.get_irdl_definition().operands) != len(
                linalg_op.body.block.args[:-1]
            ):
                # wrong number of operands, continue search
                continue
            kernel_op = op_def.make_op_from_generic(linalg_op)
            assert isinstance(kernel_op, Parsable)

            if check_kernel_equivalence(
                linalg_op.body.block, kernel_op.equivalent_region.block
            ):
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ParseLinalgBody()).rewrite_module(op)
