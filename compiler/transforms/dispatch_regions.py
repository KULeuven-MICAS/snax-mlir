from xdsl.dialects import builtin, memref, func, arith, scf, linalg
from xdsl.ir.core import Operation, Block
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from collections.abc import Iterable, Callable
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class DispatchRegionsRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        def dispatcher(
            block: Block, core_predication: builtin.i1, dispatch_rule: Callable
        ):
            ops_to_dispatch: Iterable[Operation] = []

            for op in block.walk():
                # only consider top level operations
                if op.parent is not block:
                    continue
                if dispatch_rule(op):
                    ops_to_dispatch.append(op)
                elif len(ops_to_dispatch) > 0:
                    # detach op from original region
                    for dispatch_op in ops_to_dispatch:
                        dispatch_op.detach()
                    # create scf region
                    ops_to_dispatch.append(scf.Yield())
                    # create scf
                    if_op = scf.If(core_predication, [], ops_to_dispatch)
                    rewriter.insert_op_before(if_op, op)
                    ops_to_dispatch = []

        def dispatch_to_dm(op):
            if isinstance(op, memref.CopyOp):
                return True
            return False

        def dispatch_to_compute(op):
            if isinstance(op, linalg.Generic):
                return True
            return False

        for block in [func_op.body.blocks[0]]:
            # insert core checkers
            # TODO: actually implement with snitch runtime
            check_dm_core = arith.Constant.from_int_and_width(1, 1)
            check_compute_core = arith.Constant.from_int_and_width(0, 1)

            rewriter.insert_op_at_start(check_dm_core, block)
            rewriter.insert_op_at_start(check_compute_core, block)

            dispatcher(block, check_dm_core.result, dispatch_to_dm)
            dispatcher(block, check_compute_core.result, dispatch_to_compute)

        pass


class DispatchRegions(ModulePass):
    name = "dispatch-regions"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            DispatchRegionsRewriter(), apply_recursively=False
        ).rewrite_module(module)
