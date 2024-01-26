from xdsl.dialects import builtin, linalg
from xdsl.dialects.memref import MemRefType
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.util.kernel_type import KernelType


class DispatchElementwiseMult(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        """Add a library call to a linalg generic that implements an
        elementwise multiplication. This is done on linalg.generics as
        named linalg ops do not support library calls. This makes the task
        of detecting an elementwise multiplication somewhat harder,
        but this can be more structured in future work."""

        ## conditions for library call:
        # (1) kernel type must be mul
        # (2) data type must be 1D integer memref
        # (3) dataflow must be all parallel

        if op.library_call is not None:
            return

        if KernelType.get_type(op) != KernelType.MUL:
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

        op.library_call = builtin.StringAttr("snax_hwpe_mult")

        return


class DispatchQMatMul(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        """Add a library call to a linalg generic that implements an
        quantized matmul."""

        ## conditions for library call:
        # (1) kernel type must be qmac
        # (2) data type must be 2D integer memref
        # (3) dataflow must be fit matmul flow

        if op.library_call is not None:
            return

        if KernelType.get_type(op) != KernelType.QMAC:
            return

        for inp in [x for x in op.inputs if isinstance(x.type, builtin.ShapedType)]:
            if not isinstance(inp.type, MemRefType):
                return
            if len(inp.type.get_shape()) != 2:
                return
            if not isinstance(inp.type.get_element_type(), builtin.IntegerType):
                return

        # matmul is 3d problem
        if len(op.iterator_types) != 3:
            return

        # required iterator types for valid quantized matrix multiplication
        itypes_target = [
            linalg.IteratorType.PARALLEL,
            linalg.IteratorType.PARALLEL,
            linalg.IteratorType.REDUCTION,
        ]

        if any(
            [
                itypes_target[i] != op.iterator_types.data[i].data
                for i in range(len(itypes_target))
            ]
        ):
            return

        # required input maps for valid quantized matrix multiplication
        imaps_target = [
            AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(2))),
            AffineMap(3, 0, (AffineDimExpr(2), AffineDimExpr(1))),
            AffineMap(3, 0, ()),
            AffineMap(3, 0, ()),
            AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(1))),
        ]

        if any(
            [
                imaps_target[i] != op.indexing_maps.data[i].data
                for i in range(len(imaps_target))
            ]
        ):
            return

        op.library_call = builtin.StringAttr("snax_qgemm")

        return


class DispatchKernels(ModulePass):
    """
    This pass detects integer elementwise multiplications (linalg.mul),
    and inserts a library call to snax-hwpe.
    """

    name = "dispatch-kernels"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(DispatchElementwiseMult()).rewrite_module(op)
        PatternRewriteWalker(DispatchQMatMul()).rewrite_module(op)
