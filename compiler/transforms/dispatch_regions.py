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
        def dispatcher(block: Block, core_cond: builtin.i1, dispatch_rule: Callable):
            """Helper function to create dispatches in a block. If an operation is
            dispatchable according to dispatch_rule, this function will enclose it in
            an scf.if block based on the condition core_cond"""

            # return if the dispatcher made any changes
            changes_made = False

            # group dispatchable ops to avoid two
            # scf.if statements after each other
            ops_to_dispatch: Iterable[Operation] = []

            # walk through the block
            for op in block.walk():
                # only consider top level operations
                if op.parent is not block:
                    continue

                # if op is dispatchable, add to existing list
                # of dispatchable ops to include in scf.if body
                if dispatch_rule(op):
                    ops_to_dispatch.append(op)

                # else, the scf.if body stops. in this case
                # create and insert the scf.if operation
                elif len(ops_to_dispatch) > 0:
                    # detach ops in list from original region
                    for dispatch_op in ops_to_dispatch:
                        dispatch_op.detach()
                    # add scf terminator
                    ops_to_dispatch.append(scf.Yield())
                    # create and insert scf.if op
                    if_op = scf.If(core_cond, [], ops_to_dispatch)
                    rewriter.insert_op_before(if_op, op)

                    # reset dispatchable ops list
                    ops_to_dispatch = []

                    # assert changes_made flag
                    changes_made = True

            return changes_made

        def dispatch_to_dm(op):
            """Rule to dispatch operations to the dm core:
            for now, this is only memref copy operations"""
            if isinstance(op, memref.CopyOp):
                return True
            return False

        def dispatch_to_compute(op):
            """Rule to dispatch operations to the dm core:
            for now, this is only linalg generic operations"""
            if isinstance(op, linalg.Generic):
                return True
            return False

        # find root module op of func op:
        # this is necessary to declare the external functions
        # such as is_dm_core and is_compute core at the correct place
        module_op = func_op
        while not isinstance(module_op, builtin.ModuleOp):
            module_op = module_op.parent

        for block in func_op.body.blocks:
            # Dispatch DM core ops:
            # check if core is dm core
            func_call_dm = func.Call("snrt_is_dm_core", [], [builtin.i1])

            # dispatch ops
            changes_made = dispatcher(block, func_call_dm.res[0], dispatch_to_dm)
            if changes_made:
                # Insert external function definition and function call
                func_op_dm = func.FuncOp.external("snrt_is_dm_core", [], [builtin.i1])
                SymbolTable.insert_or_update(module_op, func_op_dm)
                rewriter.insert_op_at_start(func_call_dm, block)

            # Dispatch compute core ops:
            # check if core is compute core
            func_call_compute = func.Call("snrt_is_compute_core", [], [builtin.i1])

            # dispatch ops
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


class DispatchRegions(ModulePass):
    """Transformation pass dispatch-regions. This transformation
    'dispatches' the different operations to their designated cores,
    by inserting function calls to determine the core type, and enclosing
    the dispatchable operations in an scf.if block"""

    name = "dispatch-regions"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            DispatchRegionsRewriter(), apply_recursively=False
        ).rewrite_module(module)
