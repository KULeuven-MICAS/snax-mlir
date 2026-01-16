from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from snaxc.dialects import phs


class RemoveDeadInputs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: phs.PEOp, rewriter: PatternRewriter):
        """
        Linalg-on-tensors provides an output argument in the block,
        even when there's no reduction iterator, i.e.
        when no input is really passed (it is still necessary/used for e.g. shape information).

        For PHS, this unused block argument creates an extra hardware port, which is unnecessary/unwanted.
        This pass removes the unused block argument first from the block_args,
        and then from the function_type to keep the PEOp valid.
        """

        block = op.regions[0].block
        indices: set[int] = set()

        # Remove unused inputs
        for i, block_arg in enumerate(block.args):
            if block_arg.uses.get_length() == 0:
                block.erase_arg(block_arg)
                indices.add(i)

        inputs = list(op.function_type.inputs)
        outputs = list(op.function_type.outputs)
        new_inputs = [v for i, v in enumerate(inputs) if i not in indices]
        op.function_type = builtin.FunctionType.from_lists(new_inputs, outputs)


class PhsDeadInputRemovalPass(ModulePass):
    name = "phs-dead-input-removal"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RemoveDeadInputs(), apply_recursively=False).rewrite_module(op)
