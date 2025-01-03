from collections.abc import Callable, Iterable
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable

from compiler.util.dispatching_rules import dispatch_to_compute, dispatch_to_dm


@dataclass
class DispatchRegionsRewriter(RewritePattern):
    nb_cores: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        def dispatcher(
            block: Block,
            core_cond: SSAValue | Operation,
            dispatch_rule: Callable[[Operation], bool],
        ):
            """Helper function to create dispatches in a block. If an operation is
            dispatchable according to dispatch_rule, this function will enclose it in
            an scf.if block based on the condition core_cond"""

            # return if the dispatcher made any changes
            changes_made = False

            # group dispatchable ops to avoid two
            # scf.if statements after each other
            ops_to_dispatch: Iterable[Operation] = []

            # walk through the block
            for op in block.walk(region_first=True):
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
                    ops_to_dispatch.append(scf.YieldOp())
                    # create and insert scf.if op
                    if_op = scf.IfOp(core_cond, [], ops_to_dispatch)
                    rewriter.insert_op(if_op, InsertPoint.before(op))

                    # reset dispatchable ops list
                    ops_to_dispatch = []

                    # assert changes_made flag
                    changes_made = True

            return changes_made

        ## dispatch dm core ops, insert function call
        # in dominator block if changes made

        # FIXME: currently assuming that DM core is nb_cores - 1 and compute @ index 0

        func_call = func.CallOp("snax_cluster_core_idx", [], [builtin.i32])

        # Add pin to constants attribute for function-constant-pinning pass
        constants_to_pin = builtin.ArrayAttr(
            [
                builtin.IntegerAttr.from_int_and_width(i, 32)
                for i in range(self.nb_cores)
            ]
        )
        func_call.attributes.update({"pin_to_constants": constants_to_pin})

        call_and_condition_dm = [
            func_call,
            cst_1 := arith.ConstantOp.from_int_and_width(
                self.nb_cores - 1, builtin.i32
            ),
            comparison_dm := arith.CmpiOp(func_call, cst_1, "eq"),
        ]
        # Make sure function call is only inserted once
        inserted_function_call = False
        if any(
            dispatcher(block, comparison_dm.result, dispatch_to_dm)
            for block in func_op.body.blocks
        ):
            inserted_function_call = True
            rewriter.insert_op(
                call_and_condition_dm, InsertPoint.at_start(func_op.body.blocks[0])
            )

        ## dispatch compute core ops, insert function call
        # in dominator block if changes made
        condition_compute = [
            cst_0 := arith.ConstantOp.from_int_and_width(0, builtin.i32),
            comparison_compute := arith.CmpiOp(func_call, cst_0, "eq"),
        ]
        if any(
            dispatcher(block, comparison_compute.result, dispatch_to_compute)
            for block in func_op.body.blocks
        ):
            # insert function call in dominator block (first one)
            if inserted_function_call:
                # If function call is already inserted, insert check after
                rewriter.insert_op(condition_compute, InsertPoint.after(func_call))
            else:
                # If function call is not yet inserted, insert check and function call in beginning
                rewriter.insert_op(
                    [func_call, *condition_compute],
                    InsertPoint.at_start(func_op.body.blocks[0]),
                )


class InsertFunctionDeclaration(RewritePattern):
    """Insert external function declarations of snax_cluster_core_idx if they are used in the module"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        # Check for snax cluster core idx
        if op.callee.string_value() != "snax_cluster_core_idx":
            return

        # Create Func Op
        func_op = func.FuncOp.external("snax_cluster_core_idx", [], [builtin.i32])

        # Search for ModuleOp
        module_op = op.parent
        while not isinstance(module_op, builtin.ModuleOp):
            assert module_op
            module_op = module_op.parent

        # Insert FuncOp
        SymbolTable.insert_or_update(module_op, func_op)


@dataclass(frozen=True)
class DispatchRegions(ModulePass):
    """Transformation pass dispatch-regions. This transformation
    'dispatches' the different operations to their designated cores,
    by inserting function calls to determine the core type, and enclosing
    the dispatchable operations in an scf.if block

    Emitted function calls are annotated with pin_to_constants.
    The value is based on the amount of cores in nb_cores,
    and allows one to fully specialize a top-level function body to a specific core.
    """

    name = "dispatch-regions"

    nb_cores: int = 2  # amount of cores

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            DispatchRegionsRewriter(self.nb_cores),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(
            InsertFunctionDeclaration(), apply_recursively=False
        ).rewrite_module(op)
