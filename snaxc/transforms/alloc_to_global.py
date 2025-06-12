from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, memref
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import IsTerminator, SymbolTable


class AllocToGlobal(RewritePattern):
    """
    Convert all static allocs to empty statically scheduled memref globals

    Warning: using this pattern multiple times in a lowering flow may lead
    to over
    """

    # keep track of index to give memref globals a name
    index: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter):
        # check if it is used in a terminator operation
        if not any(use.operation.has_trait(IsTerminator) for use in op.memref.uses):
            return

        # get module op
        module_op = op
        while not isinstance(module_op, builtin.ModuleOp):
            assert module_op is not None
            module_op = module_op.parent_op()

        symbol_table = module_op.get_trait(SymbolTable)
        assert symbol_table

        # create global symbol name
        global_sym_name = "_static_const_" + str(self.index)
        self.index = self.index + 1

        # check if it is not in the symbol table
        assert SymbolTable.lookup_symbol(module_op, global_sym_name) is None

        # create global
        memref_global = memref.GlobalOp.get(
            builtin.StringAttr(global_sym_name),
            op.results[0].type,
            initial_value=builtin.UnitAttr(),
        )
        SymbolTable.insert_or_update(module_op, memref_global)

        # remove all the deallocs
        deallocs = [
            user_op.operation
            for user_op in op.results[0].uses
            if isinstance(user_op.operation, memref.DeallocOp)
        ]
        for dealloc in deallocs:
            rewriter.erase_op(dealloc)

        # replace op by get global
        memref_get = memref.GetGlobalOp(global_sym_name, op.results[0].type)

        rewriter.replace_matched_op(memref_get)


@dataclass(frozen=True)
class AllocToGlobalPass(ModulePass):
    """
    Convert all present memref.allocs into statically scheduled memref.globals
    """

    name = "alloc-to-global"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AllocToGlobal()).rewrite_module(op)
