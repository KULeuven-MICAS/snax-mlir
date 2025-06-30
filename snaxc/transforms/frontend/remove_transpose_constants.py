from collections.abc import Sequence
from typing import cast

from xdsl.dialects import arith, builtin, linalg
from xdsl.ir import OpResult
from xdsl.ir.affine import AffineMap
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


class RemoveTransposeConstants(RewritePattern):
    """
    This path finds linalg generic operations that transpose a constant.
    It then constant folds this operation by transforming the weight directly.
    """

    def transpose_tuple(self, array_tuple: Sequence[int], cols: int, rows: int) -> Sequence[int]:
        # Transpose using list comprehension
        transposed_tuple = tuple(array_tuple[i + j * rows] for i in range(rows) for j in range(cols))
        return transposed_tuple

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # find transpose generics on constants and just do it directly

        # transpose op has only yield
        if not isinstance(op.body.block.first_op, linalg.YieldOp):
            return

        # check for transpose:
        if len(op.indexing_maps) != 2:
            return
        if op.indexing_maps.data[0].data != AffineMap.from_callable(lambda x, y: (y, x)):
            return
        if op.indexing_maps.data[1].data != AffineMap.from_callable(lambda x, y: (x, y)):
            return

        # is input constant?
        if not isinstance(opresult := op.inputs[0], OpResult):
            return
        if not isinstance(const_op := opresult.op, arith.ConstantOp):
            return
        if not isa((const_type := op.inputs[0].type), builtin.TensorType[builtin.IntegerType]):
            return
        if not isinstance((dense_attr := const_op.value), builtin.DenseIntOrFPElementsAttr):
            return

        # transpose const op
        transposed_data = self.transpose_tuple(cast(Sequence[int], dense_attr.get_values()), *const_type.get_shape())
        assert isa(op.outputs[0].type, builtin.TensorType[builtin.IntegerType])
        transposed_dense_attr = builtin.DenseIntOrFPElementsAttr.create_dense_int(op.outputs[0].type, transposed_data)

        # create new const_op
        new_const_op = arith.ConstantOp(transposed_dense_attr, op.outputs[0].type)

        # insert new const operation
        rewriter.insert_op(new_const_op, InsertPoint.before(const_op))

        # replace uses of transform with new const op
        op.results[0].replace_by(new_const_op.results[0])

        # delete const op and linalg op
        rewriter.erase_matched_op()
        rewriter.erase_op(const_op)
