from xdsl.dialects import builtin, linalg, arith
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.dialects.memref import MemRefType
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddLibraryCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        ## conditions for library call:
        #   (0) must not already have library call
        #   (1) 2 operands
        #   (2) both operands of type memref
        #   (3) 1D-shape
        #   (4) type integer
        #   (5) iterator type also 1D and parallel
        #   (6) region must be non reducing mult of both inputs

        if op.library_call is not None:
            return

        if len(op.inputs) != 2:
            return

        for inp in op.inputs:
            if not isinstance(inp.type, MemRefType):
                return

            if len(inp.type.get_shape()) > 1:
                return

            if not isinstance(inp.type.get_element_type(), builtin.IntegerType):
                return

        if len(op.iterator_types) != 1:
            return

        if op.iterator_types.data[0].data is not linalg.IteratorType.PARALLEL:
            return

        ## Check if operation is muli
        ## two operations: first operation is arith.muli, last operation is yield

        mult_op = op.body.block.first_op
        yield_op = op.body.block.last_op

        # last operation is linalg.yield
        if not isinstance(yield_op, linalg.YieldOp):
            return
        # first operaion is arith.muli
        if not isinstance(mult_op, arith.Muli):
            return
        # yield is result of muli
        if mult_op.result is not yield_op.arguments[0]:
            return
        # muli is based on first two args
        if not (
            op.body.block.args[0] in mult_op.operands
            and op.body.block.args[1] in mult_op.operands
        ):
            return

        op.library_call = builtin.StringAttr("snax_hwpe_mult")

        return


class DispatchElementWiseMult(ModulePass):
    """
    This pass detects integer elementwise multiplications (linalg.mul),
    and inserts a library call to snax-hwpe.
    """

    name = "dispatch-elementwise-mult"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AddLibraryCall()).rewrite_module(op)
