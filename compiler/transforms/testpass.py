from xdsl.dialects import builtin, func
from dialects import linalg_ext
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.dialects.memref import MemRefType
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddLibraryCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg_ext.Mul, rewriter: PatternRewriter):
        ## conditions for library call:
        #   (1) 2 operands
        #   (2) both operands of type memref
        #   (3) both operands of same shape (need to check this?)
        #   (4) 1D-shape
        #   (5) type integer

        # print("execute match and rewrite")

        if len(op.inputs) != 2:
            return

        for inp in op.inputs:
            if not isinstance(inp.type, MemRefType):
                return

            if len(inp.type.get_shape()) > 1:
                return

            if not isinstance(inp.type.get_element_type(), builtin.IntegerType):
                return

        op.library_call = builtin.StringAttr("snax_hwpe_mult")

        return


class LowerLinalgTofunc(RewritePattern):
    """
    Lowers linalg.mul functions marked with a library call to function calls
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg_ext.Mul, rewriter: PatternRewriter):
        if op.library_call is None:
            return

        rewriter.replace_matched_op(func.Call(op.library_call.data, op.operands, []))

        return


class AddExternalFunc(RewritePattern):
    """
    Looks for hwpe function calls and adds an external
    func call to it for LLVM to link in
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp, rewriter: PatternRewriter):
        for op in module.walk():
            print(op)
            if not isinstance(op, func.Call):
                continue
            if "snax_hwpe" not in op.callee.string_value():
                continue

            print("inserting")

            rewriter.insert_op_at_end(
                func.FuncOp.external(
                    op.callee.string_value(),
                    [arg.type for arg in op.arguments],
                    [res.type for res in op.results],
                ),
                module.body.block,
            )


class AllocateElementWiseMult(ModulePass):
    """
    This pass detects integer elementwise multiplications, and replaces them with
    an external function call hwpe_mult.
    """

    name = "allocate-elementwise-mult"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddLibraryCall(),
                    LowerLinalgTofunc(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(AddExternalFunc(), apply_recursively=False).rewrite_module(
            op
        )
