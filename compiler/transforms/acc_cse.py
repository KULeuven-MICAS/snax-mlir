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


def get_state_before(op: acc.SetupOp) -> dict[str, SSAValue]:
    """
    Walk the def-use chain up and return a dictionary representing which
    parameter is currently set to which value before `op` is encountered.
    """
    state: dict[str, SSAValue] = {}

    while op.in_state is not None:
        parent = op.in_state.owner

        # TODO: handle more stuff here later
        if not isinstance(parent, acc.SetupOp):
            raise RuntimeError("We can't handle this yet")

        for name, val in parent.iter_params():
            if name not in state:
                state[name] = val

        op = parent
    return state


class YeetSetupStuff(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # Step 1: Figure out previous state
        prev_state = get_state_before(op)

        # Step 2: Yeet out redundant stuff
        new_params: list[tuple[str, SSAValue]] = [
            (name, val) for name, val in op.iter_params() if prev_state.get(name) != val
        ]

        # Step 3: If no new params remain, yeet the whole op
        if not new_params:
            op.out_state.replace_by(op.in_state)
            rewriter.erase_matched_op()
            return

        # Step 4: If all parameters change, we are a no-op
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

    This pass analyzes the accelerator
    """

    name = "acc-cse"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(YeetSetupStuff()).rewrite_module(op)
