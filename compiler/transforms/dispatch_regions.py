from xdsl.dialects import builtin, memref, func, scf, linalg
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
from xdsl.traits import SymbolTable


class DispatchRegionsRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        def dispatcher(
            block: Block, core_predication: builtin.i1, dispatch_rule: Callable
        ):
            changes_made = False
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
                    changes_made = True
            return changes_made

        def dispatch_to_dm(op):
            if isinstance(op, memref.CopyOp):
                return True
            return False

        def dispatch_to_compute(op):
            if isinstance(op, linalg.Generic):
                return True
            return False

        # find root module op of func op:
        module_op = func_op
        while not isinstance(module_op, builtin.ModuleOp):
            module_op = module_op.parent

        for block in [func_op.body.blocks[0]]:
            func_call_dm = func.Call("snrt_is_dm_core", [], [builtin.i1])

            changes_made = dispatcher(block, func_call_dm.res[0], dispatch_to_dm)
            if changes_made:
                # Insert external function definition and function call
                func_op_dm = func.FuncOp.external("snrt_is_dm_core", [], [builtin.i1])
                SymbolTable.insert_or_update(module_op, func_op_dm)
                rewriter.insert_op_at_start(func_call_dm, block)

            func_call_compute = func.Call("snrt_is_compute_core", [], [builtin.i1])

            changes_made = dispatcher(
                block, func_call_compute.res[0], dispatch_to_compute
            )

            if changes_made:
                # Insert external function definition and function call
                func_op_compute = func.FuncOp.external(
                    "snrt_is_compute_core", [], [builtin.i1]
                )
                SymbolTable.insert_or_update(module_op, func_op_compute)
                rewriter.insert_op_at_start(func_call_compute, block)

        pass


class DispatchRegions(ModulePass):
    name = "dispatch-regions"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            DispatchRegionsRewriter(), apply_recursively=False
        ).rewrite_module(module)
