from xdsl.dialects import builtin, func, scf
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
from compiler.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm


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

        ## dispatch dm core ops, insert function call
        # in dominator block if changes made
        func_call_dm = func.Call("snax_is_dm_core", [], [builtin.i1])
        if any(
            dispatcher(block, func_call_dm.res[0], dispatch_to_dm)
            for block in func_op.body.blocks
        ):
            rewriter.insert_op_at_start(func_call_dm, func_op.body.blocks[0])

        ## dispatch compute core ops, insert function call
        # in dominator block if changes made
        func_call_compute = func.Call("snax_is_compute_core", [], [builtin.i1])
        if any(
            dispatcher(block, func_call_compute.res[0], dispatch_to_compute)
            for block in func_op.body.blocks
        ):
            # insert function call in dominator block (first one)
            rewriter.insert_op_at_start(func_call_compute, func_op.body.blocks[0])


class InsertFunctionDeclaration(RewritePattern):
    """Insert external function declarations of snax_is_compute core
    and snax_is_dm_core if they are used in the module"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module_op: builtin.ModuleOp, rewriter: PatternRewriter):
        for op in module_op.walk():
            if isinstance(op, func.Call):
                if op.callee.string_value() == "snax_is_compute_core":
                    func_op_compute = func.FuncOp.external(
                        "snax_is_compute_core", [], [builtin.i1]
                    )
                    SymbolTable.insert_or_update(module_op, func_op_compute)
                if op.callee.string_value() == "snax_is_dm_core":
                    func_op_dm = func.FuncOp.external(
                        "snax_is_dm_core", [], [builtin.i1]
                    )
                    SymbolTable.insert_or_update(module_op, func_op_dm)


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
        PatternRewriteWalker(
            InsertFunctionDeclaration(), apply_recursively=False
        ).rewrite_module(module)
