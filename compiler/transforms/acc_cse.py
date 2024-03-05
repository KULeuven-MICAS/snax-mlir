from xdsl.dialects import builtin
from xdsl.ir import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import acc
from compiler.inference.trace_acc_state import infer_state_of


class SimplifyRedundantSetupCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # Step 1: Figure out previous state
        prev_state = infer_state_of(op.in_state) if op.in_state else {}

        # Step 2: Filter setup parameters to remove calls that set the same value again
        new_params: list[tuple[str, SSAValue]] = [
            (name, val) for name, val in op.iter_params() if prev_state.get(name) != val
        ]

        # Step 3: If no new params remain, elide the whole op
        if not new_params:
            # This only happens when:
            #  1) The operation has no input state, and
            #  2) The operation has no parameters it sets
            if op.in_state is None:
                # in this case, we can't elide the operation if it's output state is
                # used otherwise we would break stuff. So we just assume that a setup
                # without parameters returns an "empty" state that assumes nothing.
                return

            op.out_state.replace_by(op.in_state)
            rewriter.erase_matched_op()
            return

        # Step 4: If all parameters change, do nothing
        if len(new_params) == len(op.param_names):
            return

        # Step 5: Replace matched op with reduced version
        rewriter.replace_matched_op(
            acc.SetupOp(
                [val for _, val in new_params],
                [param for param, _ in new_params],
                op.accelerator,
                op.in_state,
            )
        )


class AccCse(ModulePass):
    """
    Common subexpression elimination for the `acc` (accelerator) dialect.

    This pass rewrites acc dialect operations to find and simplify redundant
    setup calls.
    """

    name = "acc-cse"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(SimplifyRedundantSetupCalls()).rewrite_module(op)
