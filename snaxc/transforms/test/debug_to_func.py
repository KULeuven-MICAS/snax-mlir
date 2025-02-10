from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, memref
from xdsl.irdl import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from snaxc.dialects.test import debug


class DebugToFunc(RewritePattern):
    """Insert debugging function calls"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: debug.DebugLinalgOp, rewriter: PatternRewriter):
        ptr_a = memref.ExtractAlignedPointerAsIndexOp.get(op.op_a)
        ptr_b = memref.ExtractAlignedPointerAsIndexOp.get(op.op_b)
        ptr_c = memref.ExtractAlignedPointerAsIndexOp.get(op.op_c)

        ops_to_insert: list[Operation] = [ptr_a, ptr_b, ptr_c]

        ptr_a = arith.IndexCastOp(ptr_a, builtin.i32)
        ptr_b = arith.IndexCastOp(ptr_b, builtin.i32)
        ptr_c = arith.IndexCastOp(ptr_c, builtin.i32)

        ops_to_insert.extend([ptr_a, ptr_b, ptr_c])

        match (op.when.data, op.level.data):
            case "before", "L3":
                whenparam = 0
            case "before", "L1":
                whenparam = 1
            case "after", "L1":
                whenparam = 2
            case "after", "L3":
                whenparam = 3
            case _:
                whenparam = 5

        when = arith.ConstantOp.from_int_and_width(whenparam, 32)
        ops_to_insert.append(when)

        func_call = func.CallOp(
            f"debug_{op.debug_type.data}", [ptr_a, ptr_b, ptr_c, when], []
        )
        ops_to_insert.append(func_call)
        rewriter.replace_matched_op(ops_to_insert)

        func_decl = func.FuncOp.external(
            f"debug_{op.debug_type.data}",
            [builtin.i32, builtin.i32, builtin.i32, builtin.i32],
            [],
        )
        module_op = func_call
        while not isinstance(module_op, builtin.ModuleOp):
            x = module_op.parent_op()
            assert x is not None
            module_op = x
        SymbolTable.insert_or_update(module_op, func_decl)


class DebugToFuncPass(ModulePass):
    name = "test-debug-to-func"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(DebugToFunc()).rewrite_module(op)
