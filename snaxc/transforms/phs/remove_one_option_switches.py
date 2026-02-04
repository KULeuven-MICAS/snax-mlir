from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from snaxc.dialects import phs


class PhsRemoveOneOptionSwitches(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: phs.PEOp, rewriter: PatternRewriter):
        """
        Remove switches and choose_ops that only have one option:
        Their one option is always the chosen one!
        """
        for switch in op.get_switches():
            switchee = switch.get_user_of_unique_use()
            assert switchee is not None, (
                f"Switch does not drive one choice in the PE (got {switch.uses.get_length()} uses)"
            )
            if isinstance(switchee, phs.MuxOp):  # MuxOps are always necessary, ignore
                continue
            assert isinstance(switchee, phs.ChooseOp), "Only expect MuxOp or ChooseOp to be switched inside PEOp"
            choose_op = switchee
            choices = list(choose_op.operations())
            if len(choices) != 1:
                continue

            # Get the content of the choice out
            operation = choices[0]
            operation.operands = choose_op.data_operands
            operation.detach()
            rewriter.replace_op(choose_op, operation)
            op.remove_switch(switch)


class PhsRemoveOneOptionSwitchesPass(ModulePass):
    name = "phs-remove-one-option-switches"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(PhsRemoveOneOptionSwitches(), apply_recursively=False).rewrite_module(op)
