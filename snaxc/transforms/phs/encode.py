from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import AffineMapAttr, IndexType, ModuleOp, TensorType
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.parser import DenseIntOrFPElementsAttr, Float32Type
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from snaxc.dialects import phs


def create_generic() -> linalg.GenericOp:
    # Inputs
    tensor_type = TensorType(element_type=Float32Type(), shape=[1, 3])
    tensor_val_type = DenseIntOrFPElementsAttr.from_list(type=tensor_type, data=[1.0, 2.0, 3.1])
    tensor_cst = arith.ConstantOp(tensor_val_type)

    # Body creation
    body_block = Block(arg_types=[Float32Type(), Float32Type()])
    rhs, lhs = body_block.args
    ops = [added := arith.AddfOp(rhs, lhs), linalg.YieldOp(added)]
    body_block.add_ops(ops)
    body = Region(body_block)

    # Affine Map Creation
    d0 = AffineExpr.dimension(0)
    d1 = AffineExpr.dimension(1)

    indexing_map = AffineMap(num_dims=2, num_symbols=0, results=(d0, d1))
    im_attr = AffineMapAttr(indexing_map)

    generic = linalg.GenericOp(
        inputs=[tensor_cst.result, tensor_cst.result],
        outputs=[],
        body=body,
        indexing_maps=[im_attr, im_attr, im_attr],
        iterator_types=[
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
            linalg.IteratorTypeAttr.parallel(),
        ],
        result_types=[tensor_type],
    )
    return generic


class EncodeLinalgGeneric(RewritePattern):
    _count: dict[str, int] = {}

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        for op in linalg_op.body.ops:
            if isinstance(op, arith.FloatingPointLikeBinaryOperation):
                constant_op = arith.ConstantOp.from_int_and_width(0, IndexType())
                rewriter.insert_op(constant_op, insertion_point=InsertPoint.before(op))
                choose_op = phs.ChooseOp.from_operations(
                    self.get_id(op), op.lhs, op.rhs, constant_op, [type(op)], result_types=[Float32Type()]
                )
                rewriter.replace_op(op, choose_op)
            elif isinstance(op, linalg.YieldOp):
                yield_op = phs.YieldOp(op.operands[0])
                rewriter.replace_op(op, yield_op)
            else:
                raise NotImplementedError()
        return

    def get_id(self, op: Operation):
        """
        Use Arity to group together operations in similar encoding spaces such that e.g.:
        * the second encountered binary op will be assigned to id "_2_1"
        * the first encountered ternary op is assigned to id "_3_0"
        * ...
        """
        arity = len(op.operands)
        key = f"_{arity}"
        if key in self._count:
            current_count = self._count[key] + 1
            self._count[key] = current_count
            return key + "_" + str(current_count)
        else:
            self._count[key] = 0
            return key + "_0"


def encode_generic(module_op: ModuleOp) -> ModuleOp:
    PatternRewriteWalker(EncodeLinalgGeneric(), apply_recursively=False).rewrite_module(module_op)
    return module_op


generic = create_generic()
module_op = ModuleOp([create_generic()])
c = 10
print(c * "=" + "\nbefore:\n" + c * "=")
print(module_op)
print(c * "=" + "\nafter:\n" + c * "=")
print(encode_generic(module_op))
